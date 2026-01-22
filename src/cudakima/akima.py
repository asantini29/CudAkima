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
    
    This class provides a parallel implementation of the Akima spline interpolation algorithm. It has a CPU version as well. 
    The interpolation can be performed simultaneously on array of different sizes. This is achieved by passing as inputs multidimensional arrays padded with NaN values.
    The interpolation is performed along the last axis of the input arrays, which must have dimension equal to the one of the longest array in the batch.

    Example:
    If you have a batch of 3 arrays with 5, 6 and 7 points respectively, you can interpolate them all at once by passing a 3D array with shape=(3, 7) and padding the arrays with NaN values.:
    ```
    x = np.array([[1, 2, 3, 4, 5, np.nan, np.nan], [1, 2, 3, 4, 5, 6, np.nan], [1, 2, 3, 4, 5, 6, 7]])
    y = np.array([[1, 4, 9, 16, 25, np.nan, np.nan], [1, 4, 9, 16, 25, 36, np.nan], [1, 4, 9, 16, 25, 36, 49]])
    x_new = np.linspace(1, 5, 100)
    interpolant = AkimaInterpolant1D()
    y_new = interpolant(x_new, x, y)
    ```

    Parameters:
        use_gpu (bool): If True, the interpolation is performed on the GPU if available. If False, the interpolation is performed on the CPU.
        threadsperblock (int): The number of threads per block to use for the GPU implementation. This parameter is ignored if `use_gpu` is False.
        sanitize (bool): If True, the input data are sorted in ascending order. If False, the input data must be already sorted.
        verbose (bool): If True, print information about the interpolation process. Default is False.

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
        Check that the input data have the right shape and sort them to ensure that all the arrays are in the correct order.

        Parameters:
            x (ndarray): The x-values of the data points. Shape=(..., n).
            y (ndarray): The y-values of the data points. Shape=(..., n).

        Returns:
            x (ndarray): The sorted x values.
            y (ndarray): The y values, sorted accordingly to `x`.
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
        Skip the sorting of the input data. This is useful to save computational time when the input data are already sorted.

        Parameters:
            x (ndarray): The x-values of the data points. Shape=(..., n).
            y (ndarray): The y-values of the data points. Shape=(..., n).

        Returns:
            x (ndarray): The input `x` array.
            y (ndarray): The input `y` array.
        """

        return x, y
    
    def set_sanitize(self):
        if self.sanitize:
            self.sanitize_input = self.sort_input

        else:
            self.sanitize_input = self.pass_input
            if self.verbose:
                print('Skipping input sorting. The user must make sure that the input data are sorted in anscending order with eventual NaNs values at the end of the arrays.')
    

    def __call__(self, x_new, x, y, **kwargs):
        """
        Interpolates the values of `x_new` based on the given `x` and `y` data points.

        Parameters:
            x_new (ndarray): The new x-values to interpolate. If `self.sanitize` is `False`, they must be sorted in ascending order. Shape=(n_f,).
            x (ndarray): The x-values of the data points. If `self.sanitize` is `False`, they must be sorted in ascending order. Shape=(..., n).
                         If along some axis there are less than `n` points to interpolate, the remaining values must be NaN. 
            y (ndarray): The y-values of the data points. Shape=(..., n).
                         If along some axis there are less than `n` points to interpolate, the remaining values must be NaN.
            **kwargs: Additional keyword arguments. Added for future flexibility.

        Returns:
            ndarray: The interpolated values of `x_new`. Shape=(..., n_f).
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
        Optimized GPU interpolation with precomputed slopes and binary search
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
        Optimized CPU interpolation with precomputed slopes and parallel execution
        """
        # Step 1: Precompute all linear slopes
        linear_slopes = precompute_all_linear_slopes_cpu(x, y, nin, ngroups, nnans)
        
        # Step 2: Main interpolation with parallelization
        akima_linear_kernel_cpu(x_new, x, y, linear_slopes, 
                                         nin, ngroups, nnans, result)
    
    def cubic_interpolate_gpu(self, x_new, x, y, nin, ngroups, nnans, result):
        """
        Optimized GPU interpolation with precomputed slopes and binary search
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
        Optimized CPU interpolation with precomputed slopes and parallel execution
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
    GPU-accelerated parallel Akima Splines with multidimensional x_new support.
    
    This class extends the base AkimaInterpolant1D to handle multidimensional x_new arrays,
    where each group in the batch can have different interpolation points.
    
    Key difference from base class:
    - x_new can have shape (..., n_f) matching the batch dimensions of x and y
    - Each group gets its own set of interpolation points
    - Output shape remains (..., n_f) as before
    
    Example:
    If you have batch data with shape (M, L, K, n) and want different interpolation
    points for each group:
    ```
    x = np.random.rand(2, 3, 10)  # 2x3 batch, 10 points each
    y = np.sin(x)  # same shape
    x_new = np.random.rand(2, 3, 50)  # 2x3 batch, 50 interp points each
    
    interpolant = AkimaInterpolant1DMultiDim()
    y_new = interpolant(x_new, x, y)  # shape: (2, 3, 50)
    ```
    
    Parameters are the same as AkimaInterpolant1D.
    """
    
    def __call__(self, x_new, x, y, **kwargs):
        """
        Interpolates with multidimensional x_new arrays.

        Parameters:
            x_new (ndarray): The new x-values to interpolate. Shape=(..., n_f) where ... 
                           matches the batch dimensions of x and y.
            x (ndarray): The x-values of the data points. Shape=(..., n).
            y (ndarray): The y-values of the data points. Shape=(..., n).
            **kwargs: Additional keyword arguments.

        Returns:
            ndarray: The interpolated values. Shape=(..., n_f).
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
    Flexible Akima interpolator that automatically detects x_new dimensionality.
    
    This class automatically chooses between standard (1D x_new) and multidimensional
    x_new interpolation based on the input array shapes.
    
    Usage:
    ```
    interpolant = AkimaInterpolant1DFlexible()
    
    # Standard usage (1D x_new for all groups)
    y_new = interpolant(x_new_1d, x, y)
    
    # Multidimensional usage (different x_new for each group)  
    y_new = interpolant(x_new_multidim, x, y)
    ```
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
        Automatically dispatch to appropriate interpolation method based on x_new shape.
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