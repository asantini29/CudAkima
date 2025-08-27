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

__all__ = ['AkimaInterpolant1D']

## GPU version

@cuda.jit(device=True)
def linearslope_gpu(x, y, idx):
    dx = x[idx + 1] - x[idx]
    dy = y[idx + 1] - y[idx]
    m = dy / dx
    return m

@cuda.jit(device=True)
def splineslope_gpu(x, y, idx, start, stop):

    #! with these boundary conditions I ALWAYS NEED AT LEAST FOUR POINTS

    #boundaries conditions
    if idx == start:
        return (3*linearslope_gpu(x, y, idx) - linearslope_gpu(x, y, idx+1)) / 2
    
    elif (idx == start + 1):
        return (abs(linearslope_gpu(x, y, idx+1) - linearslope_gpu(x, y, idx)) * linearslope_gpu(x, y, idx-1) \
        + abs(linearslope_gpu(x, y, idx) - linearslope_gpu(x, y, idx-1)) * linearslope_gpu(x, y, idx)) \
        / (abs(linearslope_gpu(x, y, idx+1) - linearslope_gpu(x, y, idx)) + abs(linearslope_gpu(x, y, idx) - linearslope_gpu(x, y, idx-1)))

    elif (idx == stop - 2):
        return (abs(linearslope_gpu(x, y, idx-2) - linearslope_gpu(x, y, idx-1)) * linearslope_gpu(x, y, idx)  \
        + abs(linearslope_gpu(x, y, idx-1) - linearslope_gpu(x, y, idx)) * linearslope_gpu(x, y, idx-1)) \
        / (abs(linearslope_gpu(x, y, idx-2) - linearslope_gpu(x, y, idx-1)) + abs(linearslope_gpu(x, y, idx-1) - linearslope_gpu(x, y, idx)))

    elif idx == stop - 1:
        return (3*linearslope_gpu(x, y, idx-1) - linearslope_gpu(x, y, idx-2)) / 2
    
    mi_2 = linearslope_gpu(x, y, idx-2) # i-2
    mi_1 = linearslope_gpu(x, y, idx-1) # i-1
    mi = linearslope_gpu(x, y, idx) #i
    mi1 = linearslope_gpu(x, y, idx+1) #i+1 
            
    thr = max(abs(mi1 - mi), abs(mi - mi_1), abs(mi_1 - mi_2), -1e99) * 1e-9

    if (abs(mi - mi1) + abs(mi_2 - mi_1)) < thr:
        return (mi_1 + mi) / 2
    else:
        return (abs(mi1 - mi) * mi_1 + abs(mi_1 - mi_2) * mi) / ( abs(mi1 - mi) + abs(mi_1 - mi_2) )
    

@cuda.jit
def akima_spline_kernel_gpu(x_new, x, y, n_in, ngroups, nnans, result):
    start1 = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    increment1 = cuda.blockDim.x * cuda.gridDim.x

    start2 = cuda.blockIdx.y
    increment2 = cuda.gridDim.y

    for j in range(start2, n_in, increment2):

        start = j*ngroups
        stop = (j+1)*ngroups - nnans[j]

        if stop - start < 4:
            # use linear interpolation if there are less than 4 points 
            for i in range(start1, x_new.shape[0], increment1):
                special_index = j * x_new.shape[0] + i
                idx = -1
                if x_new[i] == x[stop - 1]:
                    result[special_index] = y[stop - 1]

                else:
                    for k in range(start, (j+1)*(ngroups) - 1):

                        if (x_new[i] >= x[k] and x_new[i] < x[k + 1]):
                            idx = k
                            break

                    mi = linearslope_gpu(x, y, idx)
                    result[special_index] = y[idx] + mi * (x_new[i] - x[idx])

        else:
            # use Akima spline interpolation
            for i in range(start1, x_new.shape[0], increment1):

                special_index = j * x_new.shape[0] + i
                idx = -1
                
                if x_new[i] == x[stop - 1]: #* deal with extrema
                    result[special_index] = y[stop - 1] 
                
                else:
                    for k in range(start, (j+1)*(ngroups) - 1):
                        
                        if (x_new[i] >= x[k] and x_new[i] < x[k + 1]):
                            idx = k
                            break
                    
                    #* Akima spline interpolation 
                    mi = linearslope_gpu(x, y, idx)
                    si = splineslope_gpu(x, y, idx, j*ngroups, stop)
                    si1 = splineslope_gpu(x, y, idx+1, j*ngroups, stop)
                
                    p0 = y[idx]

                    p1 = si

                    p2 = (3*mi - 2*si - si1) / (x[idx+1] - x[idx])

                    p3 = (si + si1 - 2*mi) / (x[idx+1] - x[idx])**2

                    result[special_index] = p0 + p1 * (x_new[i] - x[idx]) + p2 * (x_new[i] - x[idx])**2 + p3 * (x_new[i] - x[idx])**3



@numba.jit(nopython=True)
def linearslope_cpu(x, y, idx):
    dx = x[idx + 1] - x[idx]
    dy = y[idx + 1] - y[idx]
    m = dy / dx
    return m

@numba.jit(nopython=True)
def splineslope_cpu(x, y, idx, start, stop):

    #! with these boundary conditions I ALWAYS NEED AT LEAST FOUR POINTS

    #boundaries conditions
    if idx == start:
        return (3*linearslope_cpu(x, y, idx) - linearslope_cpu(x, y, idx+1)) / 2
    
    elif (idx == start + 1):
        #? mi_1 = linearslope_cpu(x, y, idx-1)
        return (abs(linearslope_cpu(x, y, idx+1) - linearslope_cpu(x, y, idx)) * linearslope_cpu(x, y, idx-1) \
        + abs(linearslope_cpu(x, y, idx) - linearslope_cpu(x, y, idx-1)) * linearslope_cpu(x, y, idx)) \
        / (abs(linearslope_cpu(x, y, idx+1) - linearslope_cpu(x, y, idx)) + abs(linearslope_cpu(x, y, idx) - linearslope_cpu(x, y, idx-1)))

    elif (idx == stop - 2):
        return (abs(linearslope_cpu(x, y, idx-2) - linearslope_cpu(x, y, idx-1)) * linearslope_cpu(x, y, idx)  \
        + abs(linearslope_cpu(x, y, idx-1) - linearslope_cpu(x, y, idx)) * linearslope_cpu(x, y, idx-1)) \
        / (abs(linearslope_cpu(x, y, idx-2) - linearslope_cpu(x, y, idx-1)) + abs(linearslope_cpu(x, y, idx-1) - linearslope_cpu(x, y, idx)))

    elif idx == stop - 1:
        return (3*linearslope_cpu(x, y, idx-1) - linearslope_cpu(x, y, idx-2)) / 2
    
    mi_2 = linearslope_cpu(x, y, idx-2) # i-2
    mi_1 = linearslope_cpu(x, y, idx-1) # i-1
    mi = linearslope_cpu(x, y, idx) #i
    mi1 = linearslope_cpu(x, y, idx+1) #i+1 
            
    thr = max(abs(mi1 - mi), abs(mi - mi_1), abs(mi_1 - mi_2), -1e99) * 1e-9

    if (abs(mi - mi1) + abs(mi_2 - mi_1)) < thr:
        return (mi_1 + mi) / 2
    else:
        return (abs(mi1 - mi) * mi_1 + abs(mi_1 - mi_2) * mi) / ( abs(mi1 - mi) + abs(mi_1 - mi_2) )


@numba.jit(nopython=True)
def akima_spline_kernel_cpu(x_new, x, y, n_in, ngroups, nnans, result):
    for j in range(n_in):

        # find the first and last non-NaN values
        start = j*ngroups
        stop = (j+1)*ngroups - nnans[j]

        if stop - start < 4:
            # use linear interpolation if there are less than 4 points
            for i in range(x_new.shape[0]):
                special_index = j * x_new.shape[0] + i
                idx = -1
                if x_new[i] == x[stop - 1]:
                    result[special_index] = y[stop - 1]

                else:
                    for k in range(start, (j+1)*(ngroups) - 1):

                        if (x_new[i] >= x[k] and x_new[i] < x[k + 1]):
                            idx = k
                            break

                    mi = linearslope_cpu(x, y, idx)
                    result[special_index] = y[idx] + mi * (x_new[i] - x[idx])            
        else:
            # use Akima spline interpolation
            for i in range(x_new.shape[0]):
                special_index = j * x_new.shape[0] + i
                idx = -1
                if x_new[i] == x[stop - 1]: #* deal with extrema
                    result[special_index] = y[stop - 1] 
                
                else:
                    for k in range(start, (j+1)*(ngroups) - 1):
                        
                        if (x_new[i] >= x[k] and x_new[i] < x[k + 1]):
                            idx = k
                            break
                    
                    #* Akima spline interpolation 
                    mi = linearslope_cpu(x, y, idx)
                    si = splineslope_cpu(x, y, idx, start, stop)
                    si1 = splineslope_cpu(x, y, idx+1, start, stop)
                
                    p0 = y[idx]

                    p1 = si

                    p2 = (3*mi - 2*si - si1) / (x[idx+1] - x[idx])

                    p3 = (si + si1 - 2*mi) / (x[idx+1] - x[idx])**2

                    result[special_index] = p0 + p1 * (x_new[i] - x[idx]) + p2 * (x_new[i] - x[idx])**2 + p3 * (x_new[i] - x[idx])**3

@cuda.jit(device=True)
def binary_search_gpu(x, target, start, stop):
    """Binary search for interval location - much faster than linear search"""
    left = start
    right = stop - 1
    
    while left <= right:
        mid = (left + right) // 2
        if mid >= stop - 1:  # Handle boundary
            return stop - 2
        if x[mid] <= target < x[mid + 1]:
            return mid
        elif target < x[mid]:
            right = mid - 1
        else:
            left = mid + 1
    
    return stop - 2  # Fallback

@cuda.jit
def precompute_slopes_kernel(x, y, slopes, nin, ngroups, nnans):
    """Precompute all linear slopes in parallel"""
    idx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    group_idx = cuda.blockIdx.y
    
    if group_idx >= nin:
        return
    
    start = group_idx * ngroups
    stop = (group_idx + 1) * ngroups - nnans[group_idx]
    
    if idx < (stop - start - 1):  # Need pairs of points
        actual_idx = start + idx
        if actual_idx < stop - 1:
            dx = x[actual_idx + 1] - x[actual_idx]
            dy = y[actual_idx + 1] - y[actual_idx]
            slopes[actual_idx] = dy / dx

@cuda.jit
def precompute_spline_slopes_kernel(x, y, linear_slopes, spline_slopes, nin, ngroups, nnans):
    """Precompute spline slopes in parallel"""
    idx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    group_idx = cuda.blockIdx.y
    
    if group_idx >= nin:
        return
    
    start = group_idx * ngroups
    stop = (group_idx + 1) * ngroups - nnans[group_idx]
    
    if idx < (stop - start) and (stop - start) >= 4:
        actual_idx = start + idx
        spline_slopes[actual_idx] = splineslope_gpu_optimized(
            x, y, linear_slopes, actual_idx, start, stop
        )

@cuda.jit(device=True)
def splineslope_gpu_optimized(x, y, linear_slopes, idx, start, stop):
    """Optimized spline slope computation using precomputed linear slopes"""
    # Boundary conditions
    if idx == start:
        return (3 * linear_slopes[idx] - linear_slopes[idx + 1]) / 2
    elif idx == start + 1:
        w1 = abs(linear_slopes[idx + 1] - linear_slopes[idx])
        w2 = abs(linear_slopes[idx] - linear_slopes[idx - 1])
        if w1 + w2 == 0:
            return (linear_slopes[idx - 1] + linear_slopes[idx]) / 2
        return (w1 * linear_slopes[idx - 1] + w2 * linear_slopes[idx]) / (w1 + w2)
    elif idx == stop - 2:
        w1 = abs(linear_slopes[idx - 2] - linear_slopes[idx - 1])
        w2 = abs(linear_slopes[idx - 1] - linear_slopes[idx])
        if w1 + w2 == 0:
            return (linear_slopes[idx - 1] + linear_slopes[idx]) / 2
        return (w1 * linear_slopes[idx] + w2 * linear_slopes[idx - 1]) / (w1 + w2)
    elif idx == stop - 1:
        return (3 * linear_slopes[idx - 1] - linear_slopes[idx - 2]) / 2
    
    # General case
    mi_2 = linear_slopes[idx - 2]
    mi_1 = linear_slopes[idx - 1] 
    mi = linear_slopes[idx]
    mi1 = linear_slopes[idx + 1]
    
    w1 = abs(mi1 - mi)
    w2 = abs(mi_1 - mi_2)
    
    thr = max(w1, abs(mi - mi_1), w2, 1e-99) * 1e-9
    
    if (w1 + w2) < thr:
        return (mi_1 + mi) / 2
    else:
        return (w1 * mi_1 + w2 * mi) / (w1 + w2)

@cuda.jit
def akima_spline_kernel_optimized(x_new, x, y, linear_slopes, spline_slopes, 
                                 nin, ngroups, nnans, result):
    """Optimized main interpolation kernel"""
    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x  # x_new index
    j = cuda.blockIdx.y  # group index
    
    if i >= x_new.shape[0] or j >= nin:
        return
    
    start = j * ngroups
    stop = (j + 1) * ngroups - nnans[j]
    special_index = j * x_new.shape[0] + i
    
    # Handle edge case
    if x_new[i] >= x[stop - 1]:
        result[special_index] = y[stop - 1]
        return
    
    # Use binary search for interval finding
    idx = binary_search_gpu(x, x_new[i], start, stop)
    
    if stop - start < 4:
        # Linear interpolation
        mi = linear_slopes[idx]
        result[special_index] = y[idx] + mi * (x_new[i] - x[idx])
    else:
        # Akima spline interpolation
        mi = linear_slopes[idx]
        si = spline_slopes[idx]
        si1 = spline_slopes[idx + 1]
        
        dx = x[idx + 1] - x[idx]
        dt = x_new[i] - x[idx]
        
        p0 = y[idx]
        p1 = si
        p2 = (3 * mi - 2 * si - si1) / dx
        p3 = (si + si1 - 2 * mi) / (dx * dx)
        
        result[special_index] = p0 + p1 * dt + p2 * dt * dt + p3 * dt * dt * dt

@numba.jit(nopython=True)
def binary_search_cpu(x, target, start, stop):
    """Binary search for interval location - CPU version"""
    left = start
    right = stop - 1
    
    while left <= right:
        mid = (left + right) // 2
        if mid >= stop - 1:
            return stop - 2
        if x[mid] <= target < x[mid + 1]:
            return mid
        elif target < x[mid]:
            right = mid - 1
        else:
            left = mid + 1
    
    return stop - 2

@numba.jit(nopython=True)
def precompute_all_linear_slopes_cpu(x, y, nin, ngroups, nnans):
    """Precompute all linear slopes for all groups"""
    total_points = nin * ngroups
    linear_slopes = np.zeros(total_points)
    
    for j in range(nin):
        start = j * ngroups
        stop = (j + 1) * ngroups - nnans[j]
        
        for i in range(start, stop - 1):
            dx = x[i + 1] - x[i]
            dy = y[i + 1] - y[i]
            linear_slopes[i] = dy / dx
    
    return linear_slopes

@numba.jit(nopython=True)
def precompute_all_spline_slopes_cpu(x, y, linear_slopes, nin, ngroups, nnans):
    """Precompute all spline slopes for all groups"""
    total_points = nin * ngroups
    spline_slopes = np.zeros(total_points)
    
    for j in range(nin):
        start = j * ngroups
        stop = (j + 1) * ngroups - nnans[j]
        
        if stop - start >= 4:  # Only compute for Akima interpolation
            for i in range(start, stop):
                spline_slopes[i] = splineslope_cpu_optimized(
                    x, y, linear_slopes, i, start, stop
                )
    
    return spline_slopes

@numba.jit(nopython=True)
def splineslope_cpu_optimized(x, y, linear_slopes, idx, start, stop):
    """Optimized spline slope computation using precomputed linear slopes"""
    # Boundary conditions
    if idx == start:
        return (3 * linear_slopes[idx] - linear_slopes[idx + 1]) / 2
    elif idx == start + 1:
        w1 = abs(linear_slopes[idx + 1] - linear_slopes[idx])
        w2 = abs(linear_slopes[idx] - linear_slopes[idx - 1])
        if w1 + w2 == 0:
            return (linear_slopes[idx - 1] + linear_slopes[idx]) / 2
        return (w1 * linear_slopes[idx - 1] + w2 * linear_slopes[idx]) / (w1 + w2)
    elif idx == stop - 2:
        w1 = abs(linear_slopes[idx - 2] - linear_slopes[idx - 1])
        w2 = abs(linear_slopes[idx - 1] - linear_slopes[idx])
        if w1 + w2 == 0:
            return (linear_slopes[idx - 1] + linear_slopes[idx]) / 2
        return (w1 * linear_slopes[idx] + w2 * linear_slopes[idx - 1]) / (w1 + w2)
    elif idx == stop - 1:
        return (3 * linear_slopes[idx - 1] - linear_slopes[idx - 2]) / 2
    
    # General case
    mi_2 = linear_slopes[idx - 2]
    mi_1 = linear_slopes[idx - 1] 
    mi = linear_slopes[idx]
    mi1 = linear_slopes[idx + 1]
    
    w1 = abs(mi1 - mi)
    w2 = abs(mi_1 - mi_2)
    
    thr = max(w1, abs(mi - mi_1), w2, 1e-99) * 1e-9
    
    if (w1 + w2) < thr:
        return (mi_1 + mi) / 2
    else:
        return (w1 * mi_1 + w2 * mi) / (w1 + w2)

@numba.jit(nopython=True, parallel=True)
def akima_spline_kernel_cpu_optimized(x_new, x, y, linear_slopes, spline_slopes, 
                                     nin, ngroups, nnans, result):
    """Optimized CPU kernel with precomputed slopes and parallel execution"""
    
    # Parallel loop over groups
    for j in numba.prange(nin):
        start = j * ngroups
        stop = (j + 1) * ngroups - nnans[j]
        
        # Sequential loop over interpolation points (inner loop)
        for i in range(x_new.shape[0]):
            special_index = j * x_new.shape[0] + i
            
            # Handle edge case
            if x_new[i] >= x[stop - 1]:
                result[special_index] = y[stop - 1]
                continue
            
            # Use binary search for interval finding
            idx = binary_search_cpu(x, x_new[i], start, stop)
            
            if stop - start < 4:
                # Linear interpolation
                mi = linear_slopes[idx]
                result[special_index] = y[idx] + mi * (x_new[i] - x[idx])
            else:
                # Akima spline interpolation
                mi = linear_slopes[idx]
                si = spline_slopes[idx]
                si1 = spline_slopes[idx + 1]
                
                dx = x[idx + 1] - x[idx]
                dt = x_new[i] - x[idx]
                
                p0 = y[idx]
                p1 = si
                p2 = (3 * mi - 2 * si - si1) / dx
                p3 = (si + si1 - 2 * mi) / (dx * dx)
                
                result[special_index] = p0 + p1 * dt + p2 * dt * dt + p3 * dt * dt * dt

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
    def __init__(self, use_gpu=True, threadsperblock=64, sanitize=False, verbose=False):
        self.verbose = verbose
        self.threadsperblock = threadsperblock
     
        if use_gpu and cuda_available:
            self.interpolate = self.interpolate_gpu
            self.xp = xp 
        else:
            if self.verbose:
                print('no CUDA or CuPy available, using CPU version')
            self.interpolate = self.interpolate_cpu
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
    
    def interpolate_gpu(self, x_new, x, y, nin, ngroups, nnans, result):
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
    
    def interpolate_cpu(self, x_new, x, y, nin, ngroups, nnans, result):
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