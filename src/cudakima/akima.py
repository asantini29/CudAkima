# -*- coding: utf-8 -*-
try:
    import cupy as xp
    cuda_available = True

except (ModuleNotFoundError, ImportError):
    import numpy as xp
    cuda_available = False
    
import numba
from numba import cuda
import numpy as np

from .kernels import (
    # GPU kernels
    linearslope_gpu,
    splineslope_gpu,
    akima_spline_kernel_gpu,
    binary_search_gpu,
    precompute_slopes_kernel,
    precompute_spline_slopes_kernel,
    splineslope_gpu_optimized,
    akima_spline_kernel_optimized,
    akima_linear_kernel,
    # CPU kernels
    linearslope_cpu,
    splineslope_cpu,
    akima_spline_kernel_cpu,
    binary_search_cpu,
    precompute_all_linear_slopes_cpu,
    precompute_all_spline_slopes_cpu,
    splineslope_cpu_optimized,
    akima_spline_kernel_cpu_optimized,
    akima_linear_kernel_cpu,
    # Multidimensional kernels
    akima_linear_kernel_gpu_multidim,
    akima_linear_kernel_cpu_multidim,
    akima_spline_kernel_gpu_multidim,
    akima_spline_kernel_cpu_multidim,
)

__all__ = ['AkimaInterpolant1D', 'AkimaInterpolant1DMultiDim', 'AkimaInterpolant1DFlexible']


# Class definitions start here

class AkimaInterpolant1D():
    """
    GPU-accelerated parallel Akima Splines.
    
    This class provides a parallel implementation of the Akima spline interpolation 
    algorithm with both GPU and CPU support. The interpolation can be performed 
    simultaneously on arrays of different sizes by passing multidimensional arrays 
    padded with NaN values. The interpolation is performed along the last axis of 
    the input arrays.

    Parameters
    ----------
    use_gpu : bool, optional
        If True, use GPU acceleration if CUDA and CuPy are available. If False or 
        if GPU is unavailable, fall back to CPU implementation. Default is True.
    threadsperblock : int, optional
        Number of threads per block for GPU kernel execution. Ignored if using CPU.
        Default is 64.
    order : {'linear', 'cubic'}, optional
        Interpolation order. 'cubic' uses Akima spline interpolation, 'linear' uses
        linear interpolation. Default is 'cubic'.
    sanitize : bool, optional
        If True, sort input data in ascending order. If False, assumes input data
        are already sorted. Set to False to improve performance when data is 
        pre-sorted. Default is False.
    verbose : bool, optional
        If True, print information about the interpolation process. Default is False.
    
    Attributes
    ----------
    xp : module
        Either numpy or cupy, depending on GPU availability
    order : str
        The interpolation order being used
    sanitize : bool
        Whether input sanitization is enabled
    threadsperblock : int
        Number of threads per GPU block
        
    Examples
    --------
    Basic usage with batch interpolation:
    
    >>> import numpy as np
    >>> from cudakima import AkimaInterpolant1D
    >>> 
    >>> # Batch of 3 arrays with different lengths (5, 6, 7 points)
    >>> x = np.array([[1, 2, 3, 4, 5, np.nan, np.nan], 
    ...               [1, 2, 3, 4, 5, 6, np.nan], 
    ...               [1, 2, 3, 4, 5, 6, 7]])
    >>> y = np.array([[1, 4, 9, 16, 25, np.nan, np.nan], 
    ...               [1, 4, 9, 16, 25, 36, np.nan], 
    ...               [1, 4, 9, 16, 25, 36, 49]])
    >>> 
    >>> # Create interpolator and interpolate
    >>> interpolant = AkimaInterpolant1D()
    >>> x_new = np.linspace(1, 5, 100)
    >>> y_new = interpolant(x_new, x, y)  # Shape: (3, 100)
    
    Using linear interpolation on CPU:
    
    >>> interpolant = AkimaInterpolant1D(use_gpu=False, order='linear')
    >>> y_new = interpolant(x_new, x, y)
    
    Notes
    -----
    - Requires at least 4 finite points for Akima spline interpolation due to 
      boundary conditions. Falls back to linear interpolation for fewer points.
    - NaN values must be placed at the end of each array in the batch.
    - All arrays in a batch must be padded to the same length (the length of the 
      longest array).

    """
    def __init__(self, use_gpu=True, threadsperblock=64, order='cubic', sanitize=False, verbose=False):
        self.verbose = verbose
        self.threadsperblock = threadsperblock

        assert order in ['linear', 'cubic'], "Order must be either 'linear' or 'cubic'."
        self.order = order

        if use_gpu and cuda_available:
            self.interpolate = self.cubic_interpolate_gpu if order == 'cubic' else self.linear_interpolate_gpu
            self.xp = xp 
        else:
            if self.verbose:
                print('no CUDA or CuPy available, using CPU version')
            self.interpolate = self.cubic_interpolate_cpu if order == 'cubic' else self.linear_interpolate_cpu
            self.xp = np 

        self.sanitize = sanitize
        self.set_sanitize()
            
    @property
    def threadsperblock(self):
        return self._threadsperblock
    
    @threadsperblock.setter
    def threadsperblock(self, value):
        self._threadsperblock = value
    
    @property
    def sanitize(self):
        return self._sanitize
    
    @sanitize.setter
    def sanitize(self, value):
        self._sanitize = value

    def sort_input(self, x, y):
        """
        Sort input data in ascending order along the last axis.

        Parameters
        ----------
        x : array_like
            The x-values of the data points. Shape (..., n).
        y : array_like
            The y-values of the data points. Shape (..., n).

        Returns
        -------
        x_sorted : ndarray
            The sorted x values.
        y_sorted : ndarray
            The y values, sorted according to x.
            
        Notes
        -----
        This method ensures that x values are in ascending order, which is required
        for the interpolation algorithm to work correctly.
        """
        x = self.xp.asarray(x)
        y = self.xp.asarray(y)
        # if x.shape != y.shape or x.shape[-1] < 4: 
        #     raise ValueError('The input arrays must have the same shape and at least 4 points.')
        
        indices_x = self.xp.argsort(x, axis=-1)
        x = self.xp.take_along_axis(x, indices_x, axis=-1)
        y = self.xp.take_along_axis(y, indices_x, axis=-1)

        return x, y
    

    def pass_input(self, x, y):
        """
        Pass input data through without sorting.
        
        This is used when sanitize=False to skip the sorting step for performance.

        Parameters
        ----------
        x : array_like
            The x-values of the data points. Shape (..., n).
        y : array_like
            The y-values of the data points. Shape (..., n).

        Returns
        -------
        x : ndarray
            The input x array (unchanged).
        y : ndarray
            The input y array (unchanged).
            
        Notes
        -----
        When using this method, the user must ensure that input data are already
        sorted in ascending order with NaN values at the end.
        """

        return x, y
    
    def set_sanitize(self):
        if self.sanitize:
            self.sanitize_input = self.sort_input

        else:
            self.sanitize_input = self.pass_input
            if self.verbose:
                print('Skipping input sorting. The user must make sure that the input data are sorted in ascending order with eventual NaNs values at the end of the arrays.')
    

    def __call__(self, x_new, x, y, **kwargs):
        """
        Interpolate at new x-values.

        Parameters
        ----------
        x_new : array_like, shape (n_f,)
            New x-values at which to interpolate. If sanitize=False, must be sorted
            in ascending order.
        x : array_like, shape (..., n)
            X-values of the data points. If sanitize=False, must be sorted in 
            ascending order along the last axis. Arrays with fewer than n points
            should be padded with NaN values at the end.
        y : array_like, shape (..., n)
            Y-values of the data points. Shape must match x. Arrays with fewer 
            than n points should be padded with NaN values at the end.
        **kwargs : dict, optional
            Additional keyword arguments (reserved for future use).

        Returns
        -------
        y_new : ndarray, shape (..., n_f)
            Interpolated values at x_new positions.
            
        Examples
        --------
        >>> import numpy as np
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> y = np.array([1, 4, 9, 16, 25])
        >>> interpolant = AkimaInterpolant1D()
        >>> x_new = np.array([1.5, 2.5, 3.5])
        >>> y_new = interpolant(x_new, x, y)
        
        Notes
        -----
        The method automatically handles batches of different-length arrays by
        using NaN padding. The interpolation is performed in parallel across
        all arrays in the batch.
        """
        x, y = self.sanitize_input(x, y)

        ngroups = x.shape[-1]
        shape_out = x.shape[:-1] + x_new.shape

        x = x.reshape(-1, ngroups, order='C')
        y = y.reshape(-1, ngroups, order='C')

        nf = x_new.shape[0]
        nin = x.shape[0]
        result = self.xp.zeros(nin * nf)

        x_flat = x.flatten()
        y_flat = y.flatten()
        nnans = self.xp.count_nonzero(self.xp.isnan(x), axis=1)

        self.interpolate(x_new, x_flat, y_flat, nin, ngroups, nnans, result)
        result = result.reshape(shape_out, order='C')
    
        return result
    
    def linear_interpolate_gpu(self, x_new, x, y, nin, ngroups, nnans, result):
        """
        GPU implementation of linear interpolation with precomputed slopes.
        
        This method performs optimized linear interpolation on the GPU using
        precomputed slopes and binary search for interval location.

        Parameters
        ----------
        x_new : array_like
            New x-values for interpolation
        x : array_like
            Flattened x data
        y : array_like
            Flattened y data
        nin : int
            Number of groups
        ngroups : int
            Maximum points per group
        nnans : array_like
            Number of NaN values per group
        result : array_like
            Output array for interpolated values
        """
        # Allocate temporary arrays for precomputed slopes
        total_points = nin * ngroups
        linear_slopes = self.xp.zeros(total_points)
        
        # Grid configuration for preprocessing
        max_points_per_group = ngroups
        precompute_threads = min(self.threadsperblock, max_points_per_group)
        precompute_blocks_x = (max_points_per_group + precompute_threads - 1) // precompute_threads
        precompute_grid = (precompute_blocks_x, nin, 1)
        
        # Step 1: Precompute linear slopes
        precompute_slopes_kernel[precompute_grid, precompute_threads](
            x, y, linear_slopes, nin, ngroups, nnans
        )
        
        
        # Step 2: Main interpolation with optimized memory access
        interp_threads = min(self.threadsperblock, x_new.shape[0])
        interp_blocks_x = (x_new.shape[0] + interp_threads - 1) // interp_threads
        interp_grid = (interp_blocks_x, nin, 1)
        
        akima_linear_kernel[interp_grid, interp_threads](
            x_new, x, y, linear_slopes, nin, ngroups, nnans, result
        )
    
    def linear_interpolate_cpu(self, x_new, x, y, nin, ngroups, nnans, result):
        """
        CPU implementation of linear interpolation with precomputed slopes.
        
        This method performs optimized linear interpolation on the CPU using
        precomputed slopes, binary search, and parallel execution.

        Parameters
        ----------
        x_new : array_like
            New x-values for interpolation
        x : array_like
            Flattened x data
        y : array_like
            Flattened y data
        nin : int
            Number of groups
        ngroups : int
            Maximum points per group
        nnans : array_like
            Number of NaN values per group
        result : array_like
            Output array for interpolated values
        """
        # Step 1: Precompute all linear slopes
        linear_slopes = precompute_all_linear_slopes_cpu(x, y, nin, ngroups, nnans)
        
        # Step 2: Main interpolation with parallelization
        akima_linear_kernel_cpu(x_new, x, y, linear_slopes, 
                                         nin, ngroups, nnans, result)
    
    def cubic_interpolate_gpu(self, x_new, x, y, nin, ngroups, nnans, result):
        """
        GPU implementation of Akima spline interpolation with precomputed slopes.
        
        This method performs optimized Akima spline interpolation on the GPU using
        precomputed linear and spline slopes with binary search for interval location.

        Parameters
        ----------
        x_new : array_like
            New x-values for interpolation
        x : array_like
            Flattened x data
        y : array_like
            Flattened y data
        nin : int
            Number of groups
        ngroups : int
            Maximum points per group
        nnans : array_like
            Number of NaN values per group
        result : array_like
            Output array for interpolated values
        """
        # Allocate temporary arrays for precomputed slopes
        total_points = nin * ngroups
        linear_slopes = self.xp.zeros(total_points)
        spline_slopes = self.xp.zeros(total_points)
        
        # Grid configuration for preprocessing
        max_points_per_group = ngroups
        precompute_threads = min(self.threadsperblock, max_points_per_group)
        precompute_blocks_x = (max_points_per_group + precompute_threads - 1) // precompute_threads
        precompute_grid = (precompute_blocks_x, nin, 1)
        
        # Step 1: Precompute linear slopes
        precompute_slopes_kernel[precompute_grid, precompute_threads](
            x, y, linear_slopes, nin, ngroups, nnans
        )
        
        # Step 2: Precompute spline slopes (depends on linear slopes)
        precompute_spline_slopes_kernel[precompute_grid, precompute_threads](
            x, y, linear_slopes, spline_slopes, nin, ngroups, nnans
        )
        
        # Step 3: Main interpolation with optimized memory access
        interp_threads = min(self.threadsperblock, x_new.shape[0])
        interp_blocks_x = (x_new.shape[0] + interp_threads - 1) // interp_threads
        interp_grid = (interp_blocks_x, nin, 1)
        
        akima_spline_kernel_optimized[interp_grid, interp_threads](
            x_new, x, y, linear_slopes, spline_slopes, nin, ngroups, nnans, result
        )
    
    def cubic_interpolate_cpu(self, x_new, x, y, nin, ngroups, nnans, result):
        """
        CPU implementation of Akima spline interpolation with precomputed slopes.
        
        This method performs optimized Akima spline interpolation on the CPU using
        precomputed linear and spline slopes with binary search and parallel execution.

        Parameters
        ----------
        x_new : array_like
            New x-values for interpolation
        x : array_like
            Flattened x data
        y : array_like
            Flattened y data
        nin : int
            Number of groups
        ngroups : int
            Maximum points per group
        nnans : array_like
            Number of NaN values per group
        result : array_like
            Output array for interpolated values
        """
        # Step 1: Precompute all linear slopes
        linear_slopes = precompute_all_linear_slopes_cpu(x, y, nin, ngroups, nnans)
        
        # Step 2: Precompute all spline slopes
        spline_slopes = precompute_all_spline_slopes_cpu(x, y, linear_slopes, nin, ngroups, nnans)
        
        # Step 3: Main interpolation with parallelization
        akima_spline_kernel_cpu_optimized(x_new, x, y, linear_slopes, spline_slopes, 
                                         nin, ngroups, nnans, result)
        
class AkimaInterpolant1DMultiDim(AkimaInterpolant1D):
    """
    Akima spline interpolator with multidimensional x_new support.
    
    This class extends AkimaInterpolant1D to handle multidimensional x_new arrays,
    where each group in the batch can have its own set of interpolation points.
    
    The key difference from the base class is that x_new can have shape (..., n_f)
    matching the batch dimensions of x and y, allowing different interpolation
    points for each group.
    
    Parameters
    ----------
    use_gpu : bool, optional
        If True, use GPU acceleration if available. Default is True.
    threadsperblock : int, optional
        Number of threads per block for GPU execution. Default is 64.
    order : {'linear', 'cubic'}, optional
        Interpolation order. Default is 'cubic'.
    sanitize : bool, optional
        If True, sort input data. Default is False.
    verbose : bool, optional
        If True, print information. Default is False.
    
    Examples
    --------
    With multidimensional x_new where each group has different interpolation points:
    
    >>> import numpy as np
    >>> from cudakima import AkimaInterpolant1DMultiDim
    >>> 
    >>> # Batch data with shape (2, 3, 10)
    >>> x = np.random.rand(2, 3, 10)
    >>> y = np.sin(x)
    >>> 
    >>> # Different interpolation points for each group (2, 3, 50)
    >>> x_new = np.random.rand(2, 3, 50)
    >>> 
    >>> interpolant = AkimaInterpolant1DMultiDim()
    >>> y_new = interpolant(x_new, x, y)  # Shape: (2, 3, 50)
    
    Notes
    -----
    - x_new must have the same batch dimensions as x and y
    - Each group gets its own set of interpolation points
    - Inherits all methods and attributes from AkimaInterpolant1D
    
    See Also
    --------
    AkimaInterpolant1D : Base class with 1D x_new
    AkimaInterpolant1DFlexible : Automatically chooses between 1D and multidimensional modes
    """
    
    def __call__(self, x_new, x, y, **kwargs):
        """
        Interpolate with multidimensional x_new arrays.

        Parameters
        ----------
        x_new : array_like, shape (..., n_f)
            New x-values at which to interpolate. Batch dimensions (...) must 
            match those of x and y.
        x : array_like, shape (..., n)
            X-values of the data points.
        y : array_like, shape (..., n)
            Y-values of the data points.
        **kwargs : dict, optional
            Additional keyword arguments.

        Returns
        -------
        y_new : ndarray, shape (..., n_f)
            Interpolated values.
            
        Raises
        ------
        ValueError
            If x_new batch dimensions don't match x and y batch dimensions.
        """
        # Validate input shapes
        if x.shape[:-1] != y.shape[:-1]:
            raise ValueError("x and y must have the same batch dimensions")
        
        if x_new.shape[:-1] != x.shape[:-1]:
            raise ValueError("x_new batch dimensions must match x and y batch dimensions")
        
        x, y = self.sanitize_input(x, y)
        
        # If sanitize is enabled, we also need to sort x_new accordingly
        if self.sanitize:
            x_new = self._sort_x_new_with_xy(x_new, x, y)
        
        ngroups = x.shape[-1]
        n_interp = x_new.shape[-1]
        batch_shape = x.shape[:-1]
        
        # Reshape for processing
        x = x.reshape(-1, ngroups, order='C')
        y = y.reshape(-1, ngroups, order='C')
        x_new = x_new.reshape(-1, n_interp, order='C')
        
        nin = x.shape[0]
        result = self.xp.zeros(nin * n_interp)
        
        x_flat = x.flatten()
        y_flat = y.flatten()
        nnans = self.xp.count_nonzero(self.xp.isnan(x), axis=1)
        
        self.interpolate_multidim(x_new, x_flat, y_flat, nin, ngroups, nnans, result)
        
        # Reshape result to match expected output shape
        result = result.reshape(batch_shape + (n_interp,), order='C')
        
        return result
    
    def _sort_x_new_with_xy(self, x_new, x_sorted, y_sorted):
        """
        Sort x_new arrays to match the sorting applied to x and y.
        This ensures consistency when sanitize=True.
        """
        # For multidimensional x_new, we sort each group independently
        x_new = self.xp.asarray(x_new)
        
        # Sort x_new along the last axis for each group
        indices_x_new = self.xp.argsort(x_new, axis=-1)
        x_new_sorted = self.xp.take_along_axis(x_new, indices_x_new, axis=-1)
        
        return x_new_sorted
    
    def interpolate_multidim(self, x_new, x, y, nin, ngroups, nnans, result):
        """Dispatch to appropriate multidimensional interpolation method"""
        if hasattr(self, 'cubic_interpolate_gpu') and self.xp != np:
            self.cubic_interpolate_gpu_multidim(x_new, x, y, nin, ngroups, nnans, result) if self.order == 'cubic' else self.linear_interpolate_gpu_multidim(x_new, x, y, nin, ngroups, nnans, result)
        else:
            self.cubic_interpolate_cpu_multidim(x_new, x, y, nin, ngroups, nnans, result) if self.order == 'cubic' else self.linear_interpolate_cpu_multidim(x_new, x, y, nin, ngroups, nnans, result)

    def linear_interpolate_gpu_multidim(self, x_new, x, y, nin, ngroups, nnans, result):
        """
        GPU interpolation for multidimensional x_new arrays
        """
        # Allocate temporary arrays for precomputed slopes
        total_points = nin * ngroups
        linear_slopes = self.xp.zeros(total_points)
        
        # Grid configuration for preprocessing (same as before)
        max_points_per_group = ngroups
        precompute_threads = min(self.threadsperblock, max_points_per_group)
        precompute_blocks_x = (max_points_per_group + precompute_threads - 1) // precompute_threads
        precompute_grid = (precompute_blocks_x, nin, 1)
        
        # Step 1: Precompute linear slopes
        precompute_slopes_kernel[precompute_grid, precompute_threads](
            x, y, linear_slopes, nin, ngroups, nnans
        )
        
        # Step 2: Main interpolation with multidimensional x_new
        n_interp = x_new.shape[1]
        interp_threads = min(self.threadsperblock, n_interp)
        interp_blocks_x = (n_interp + interp_threads - 1) // interp_threads
        interp_grid = (interp_blocks_x, nin, 1)
        
        akima_linear_kernel_gpu_multidim[interp_grid, interp_threads](
            x_new, x, y, linear_slopes, nin, ngroups, nnans, result
        )
    
    def linear_interpolate_cpu_multidim(self, x_new, x, y, nin, ngroups, nnans, result):
        """
        CPU interpolation for multidimensional x_new arrays
        """
        # Step 1: Precompute all linear slopes
        linear_slopes = precompute_all_linear_slopes_cpu(x, y, nin, ngroups, nnans)
        
        # Step 2: Main interpolation with multidimensional x_new
        akima_linear_kernel_cpu_multidim(x_new, x, y, linear_slopes,
                                        nin, ngroups, nnans, result)
    
    def cubic_interpolate_gpu_multidim(self, x_new, x, y, nin, ngroups, nnans, result):
        """
        GPU interpolation for multidimensional x_new arrays
        """
        # Allocate temporary arrays for precomputed slopes
        total_points = nin * ngroups
        linear_slopes = self.xp.zeros(total_points)
        spline_slopes = self.xp.zeros(total_points)
        
        # Grid configuration for preprocessing (same as before)
        max_points_per_group = ngroups
        precompute_threads = min(self.threadsperblock, max_points_per_group)
        precompute_blocks_x = (max_points_per_group + precompute_threads - 1) // precompute_threads
        precompute_grid = (precompute_blocks_x, nin, 1)
        
        # Step 1: Precompute linear slopes
        precompute_slopes_kernel[precompute_grid, precompute_threads](
            x, y, linear_slopes, nin, ngroups, nnans
        )
        
        # Step 2: Precompute spline slopes
        precompute_spline_slopes_kernel[precompute_grid, precompute_threads](
            x, y, linear_slopes, spline_slopes, nin, ngroups, nnans
        )
        
        # Step 3: Main interpolation with multidimensional x_new
        n_interp = x_new.shape[1]
        interp_threads = min(self.threadsperblock, n_interp)
        interp_blocks_x = (n_interp + interp_threads - 1) // interp_threads
        interp_grid = (interp_blocks_x, nin, 1)
        
        akima_spline_kernel_gpu_multidim[interp_grid, interp_threads](
            x_new, x, y, linear_slopes, spline_slopes, nin, ngroups, nnans, result
        )
    
    def cubic_interpolate_cpu_multidim(self, x_new, x, y, nin, ngroups, nnans, result):
        """
        CPU interpolation for multidimensional x_new arrays
        """
        # Step 1: Precompute all linear slopes
        linear_slopes = precompute_all_linear_slopes_cpu(x, y, nin, ngroups, nnans)
        
        # Step 2: Precompute all spline slopes
        spline_slopes = precompute_all_spline_slopes_cpu(x, y, linear_slopes, nin, ngroups, nnans)
        
        # Step 3: Main interpolation with multidimensional x_new
        akima_spline_kernel_cpu_multidim(x_new, x, y, linear_slopes, spline_slopes, 
                                        nin, ngroups, nnans, result)

class AkimaInterpolant1DFlexible(AkimaInterpolant1D):
    """
    Flexible Akima interpolator with automatic mode detection.
    
    This class automatically detects whether x_new is 1D (broadcast across all groups)
    or multidimensional (different for each group) and dispatches to the appropriate
    interpolation method.
    
    This provides a unified interface that handles both use cases seamlessly.
    
    Parameters
    ----------
    use_gpu : bool, optional
        If True, use GPU acceleration if available. Default is True.
    threadsperblock : int, optional
        Number of threads per block for GPU execution. Default is 64.
    order : {'linear', 'cubic'}, optional
        Interpolation order. Default is 'cubic'.
    sanitize : bool, optional
        If True, sort input data. Default is False.
    verbose : bool, optional
        If True, print information. Default is False.
    
    Examples
    --------
    Works seamlessly with both 1D and multidimensional x_new:
    
    >>> import numpy as np
    >>> from cudakima import AkimaInterpolant1DFlexible
    >>> 
    >>> interpolant = AkimaInterpolant1DFlexible()
    >>> 
    >>> # Standard usage (1D x_new broadcast to all groups)
    >>> x = np.random.rand(10, 20)
    >>> y = np.sin(x)
    >>> x_new_1d = np.linspace(0, 1, 100)
    >>> y_new = interpolant(x_new_1d, x, y)  # Shape: (10, 100)
    >>> 
    >>> # Multidimensional usage (different x_new for each group)  
    >>> x_new_multi = np.random.rand(10, 100)
    >>> y_new = interpolant(x_new_multi, x, y)  # Shape: (10, 100)
    
    Notes
    -----
    The class automatically determines the mode based on x_new shape:
    - If x_new.shape[:-1] == x.shape[:-1], uses multidimensional mode
    - Otherwise, uses standard mode (broadcasts x_new)
    
    See Also
    --------
    AkimaInterpolant1D : Base class for standard interpolation
    AkimaInterpolant1DMultiDim : Multidimensional interpolation
    """

    @property
    def multidim_interpolator(self):
        """
        Create and return the multidimensional Akima interpolator.
        """
        return AkimaInterpolant1DMultiDim(
            use_gpu=(self.xp != np), 
            threadsperblock=self.threadsperblock,
            order=self.order,
            sanitize=self.sanitize,
            verbose=self.verbose
        )

    def __call__(self, x_new, x, y, **kwargs):
        """
        Automatically dispatch to appropriate interpolation method.
        
        Determines whether to use standard or multidimensional interpolation
        based on the shape of x_new.

        Parameters
        ----------
        x_new : array_like
            New x-values for interpolation. Can be either:
            - 1D array (n_f,) - broadcast to all groups
            - Multidimensional (..., n_f) - different for each group
        x : array_like, shape (..., n)
            X-values of the data points.
        y : array_like, shape (..., n)
            Y-values of the data points.
        **kwargs : dict, optional
            Additional keyword arguments.

        Returns
        -------
        y_new : ndarray
            Interpolated values. Shape depends on x_new:
            - If x_new is 1D: shape (..., n_f)
            - If x_new is multidimensional: shape (..., n_f)
        """
        x_new = self.xp.asarray(x_new)
        x = self.xp.asarray(x)
        y = self.xp.asarray(y)
        
        # Check if x_new is multidimensional (matches batch dims of x,y)
        if len(x_new.shape) > 1 and x_new.shape[:-1] == x.shape[:-1]:
            # Use multidimensional interpolation
            return self._call_multidim(x_new, x, y, **kwargs)
        else:
            # Use standard interpolation (broadcast x_new across all groups)
            return super().__call__(x_new, x, y, **kwargs)
    
    def _call_multidim(self, x_new, x, y, **kwargs):
        """Internal method for multidimensional interpolation"""    
        
        return self.multidim_interpolator(x_new, x, y, **kwargs)