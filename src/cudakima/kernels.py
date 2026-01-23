# -*- coding: utf-8 -*-
"""
Numba kernels and helper functions for Akima spline interpolation.

This module contains GPU and CPU kernels for computing Akima spline interpolations,
including helper functions for slope calculations and binary search.
"""

try:
    import cupy as xp
    cuda_available = True
except (ModuleNotFoundError, ImportError):
    import numpy as xp
    cuda_available = False
    
import numba
from numba import cuda
import numpy as np


## GPU version

@cuda.jit(device=True)
def linearslope_gpu(x, y, idx):
    """
    Compute the linear slope between two consecutive points on GPU.
    
    This device function calculates the slope (derivative) between points
    at indices idx and idx+1.
    
    Parameters
    ----------
    x : array_like
        X-coordinates of data points
    y : array_like
        Y-coordinates of data points
    idx : int
        Index of the first point in the pair
        
    Returns
    -------
    float
        Linear slope (dy/dx) between points idx and idx+1
    """
    dx = x[idx + 1] - x[idx]
    dy = y[idx + 1] - y[idx]
    m = dy / dx
    return m

@cuda.jit(device=True)
def splineslope_gpu(x, y, idx, start, stop):
    """
    Compute the Akima spline slope at a given point on GPU.
    
    This device function calculates the spline slope using the Akima algorithm,
    which provides smooth interpolation with special handling for boundary conditions.
    Requires at least 4 points due to boundary conditions.
    
    Parameters
    ----------
    x : array_like
        X-coordinates of data points
    y : array_like
        Y-coordinates of data points
    idx : int
        Index at which to compute the spline slope
    start : int
        Starting index of the data segment
    stop : int
        Ending index of the data segment (exclusive)
        
    Returns
    -------
    float
        Akima spline slope at the specified point
        
    Notes
    -----
    The Akima spline uses weighted averages of neighboring slopes with special
    boundary conditions at the endpoints and near-endpoints.
    """

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
    """
    GPU kernel for Akima spline interpolation.
    
    This kernel performs parallel Akima spline interpolation across multiple
    groups of data. Falls back to linear interpolation for groups with less
    than 4 valid points.
    
    Parameters
    ----------
    x_new : array_like, shape (n_f,)
        New x-coordinates at which to interpolate
    x : array_like, shape (n_in * ngroups,)
        Flattened array of x-coordinates for all groups
    y : array_like, shape (n_in * ngroups,)
        Flattened array of y-coordinates for all groups
    n_in : int
        Number of groups to interpolate
    ngroups : int
        Maximum number of points per group (including NaNs)
    nnans : array_like, shape (n_in,)
        Number of NaN values at the end of each group
    result : array_like, shape (n_in * n_f,)
        Output array for interpolated values
    """
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
    """
    Compute the linear slope between two consecutive points on CPU.
    
    This function calculates the slope (derivative) between points
    at indices idx and idx+1.
    
    Parameters
    ----------
    x : array_like
        X-coordinates of data points
    y : array_like
        Y-coordinates of data points
    idx : int
        Index of the first point in the pair
        
    Returns
    -------
    float
        Linear slope (dy/dx) between points idx and idx+1
    """
    dx = x[idx + 1] - x[idx]
    dy = y[idx + 1] - y[idx]
    m = dy / dx
    return m

@numba.jit(nopython=True)
def splineslope_cpu(x, y, idx, start, stop):
    """
    Compute the Akima spline slope at a given point on CPU.
    
    This function calculates the spline slope using the Akima algorithm,
    which provides smooth interpolation with special handling for boundary conditions.
    Requires at least 4 points due to boundary conditions.
    
    Parameters
    ----------
    x : array_like
        X-coordinates of data points
    y : array_like
        Y-coordinates of data points
    idx : int
        Index at which to compute the spline slope
    start : int
        Starting index of the data segment
    stop : int
        Ending index of the data segment (exclusive)
        
    Returns
    -------
    float
        Akima spline slope at the specified point
        
    Notes
    -----
    The Akima spline uses weighted averages of neighboring slopes with special
    boundary conditions at the endpoints and near-endpoints.
    """

    #! with these boundary conditions I ALWAYS NEED AT LEAST FOUR POINTS

    #boundaries conditions
    if idx == start:
        return (3*linearslope_cpu(x, y, idx) - linearslope_cpu(x, y, idx+1)) / 2
    
    elif (idx == start + 1):
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
    """
    CPU kernel for Akima spline interpolation.
    
    This function performs Akima spline interpolation across multiple groups
    of data. Falls back to linear interpolation for groups with less than 4
    valid points.
    
    Parameters
    ----------
    x_new : array_like, shape (n_f,)
        New x-coordinates at which to interpolate
    x : array_like, shape (n_in * ngroups,)
        Flattened array of x-coordinates for all groups
    y : array_like, shape (n_in * ngroups,)
        Flattened array of y-coordinates for all groups
    n_in : int
        Number of groups to interpolate
    ngroups : int
        Maximum number of points per group (including NaNs)
    nnans : array_like, shape (n_in,)
        Number of NaN values at the end of each group
    result : array_like, shape (n_in * n_f,)
        Output array for interpolated values
    """
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
    """
    Binary search for interval location on GPU.
    
    This device function performs binary search to find the interval [x[i], x[i+1])
    that contains the target value. Much faster than linear search for large arrays.
    
    Parameters
    ----------
    x : array_like
        Sorted array of x-coordinates
    target : float
        Value to search for
    start : int
        Starting index of the search range
    stop : int
        Ending index of the search range (exclusive)
        
    Returns
    -------
    int
        Index i such that x[i] <= target < x[i+1]
    """
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
    """
    GPU kernel to precompute all linear slopes in parallel.
    
    This kernel computes linear slopes between consecutive points for all groups,
    storing them for later use in interpolation kernels.
    
    Parameters
    ----------
    x : array_like, shape (nin * ngroups,)
        Flattened array of x-coordinates for all groups
    y : array_like, shape (nin * ngroups,)
        Flattened array of y-coordinates for all groups
    slopes : array_like, shape (nin * ngroups,)
        Output array for computed linear slopes
    nin : int
        Number of groups
    ngroups : int
        Maximum number of points per group
    nnans : array_like, shape (nin,)
        Number of NaN values at the end of each group
    """
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
    """
    GPU kernel to precompute Akima spline slopes in parallel.
    
    This kernel computes Akima spline slopes at all points using precomputed
    linear slopes, storing them for later use in interpolation kernels.
    
    Parameters
    ----------
    x : array_like, shape (nin * ngroups,)
        Flattened array of x-coordinates for all groups
    y : array_like, shape (nin * ngroups,)
        Flattened array of y-coordinates for all groups
    linear_slopes : array_like, shape (nin * ngroups,)
        Precomputed linear slopes between consecutive points
    spline_slopes : array_like, shape (nin * ngroups,)
        Output array for computed Akima spline slopes
    nin : int
        Number of groups
    ngroups : int
        Maximum number of points per group
    nnans : array_like, shape (nin,)
        Number of NaN values at the end of each group
    """
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
    """
    Optimized Akima spline slope computation using precomputed linear slopes on GPU.
    
    This device function computes Akima spline slopes more efficiently by reusing
    precomputed linear slopes instead of recalculating them.
    
    Parameters
    ----------
    x : array_like
        X-coordinates of data points
    y : array_like
        Y-coordinates of data points
    linear_slopes : array_like
        Precomputed linear slopes between consecutive points
    idx : int
        Index at which to compute the spline slope
    start : int
        Starting index of the data segment
    stop : int
        Ending index of the data segment (exclusive)
        
    Returns
    -------
    float
        Akima spline slope at the specified point
    """
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
    """
    Optimized GPU kernel for Akima spline interpolation with precomputed slopes.
    
    This kernel performs fast parallel Akima spline interpolation using
    precomputed linear and spline slopes. Uses binary search for efficient
    interval location.
    
    Parameters
    ----------
    x_new : array_like, shape (n_f,)
        New x-coordinates at which to interpolate
    x : array_like, shape (nin * ngroups,)
        Flattened array of x-coordinates for all groups
    y : array_like, shape (nin * ngroups,)
        Flattened array of y-coordinates for all groups
    linear_slopes : array_like, shape (nin * ngroups,)
        Precomputed linear slopes between consecutive points
    spline_slopes : array_like, shape (nin * ngroups,)
        Precomputed Akima spline slopes at all points
    nin : int
        Number of groups to interpolate
    ngroups : int
        Maximum number of points per group
    nnans : array_like, shape (nin,)
        Number of NaN values at the end of each group
    result : array_like, shape (nin * n_f,)
        Output array for interpolated values
    """
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

@cuda.jit
def akima_linear_kernel(x_new, x, y, linear_slopes,nin, ngroups, nnans, result):
    """
    GPU kernel for linear interpolation with precomputed slopes.
    
    This kernel performs fast parallel linear interpolation using precomputed
    slopes and binary search for interval location.
    
    Parameters
    ----------
    x_new : array_like, shape (n_f,)
        New x-coordinates at which to interpolate
    x : array_like, shape (nin * ngroups,)
        Flattened array of x-coordinates for all groups
    y : array_like, shape (nin * ngroups,)
        Flattened array of y-coordinates for all groups
    linear_slopes : array_like, shape (nin * ngroups,)
        Precomputed linear slopes between consecutive points
    nin : int
        Number of groups to interpolate
    ngroups : int
        Maximum number of points per group
    nnans : array_like, shape (nin,)
        Number of NaN values at the end of each group
    result : array_like, shape (nin * n_f,)
        Output array for interpolated values
    """
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
    
    # Linear interpolation
    mi = linear_slopes[idx]
    result[special_index] = y[idx] + mi * (x_new[i] - x[idx])

@numba.jit(nopython=True)
def binary_search_cpu(x, target, start, stop):
    """
    Binary search for interval location on CPU.
    
    This function performs binary search to find the interval [x[i], x[i+1])
    that contains the target value. Much faster than linear search for large arrays.
    
    Parameters
    ----------
    x : array_like
        Sorted array of x-coordinates
    target : float
        Value to search for
    start : int
        Starting index of the search range
    stop : int
        Ending index of the search range (exclusive)
        
    Returns
    -------
    int
        Index i such that x[i] <= target < x[i+1]
    """
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
    """
    Precompute all linear slopes for all groups on CPU.
    
    This function computes linear slopes between consecutive points for all groups,
    storing them for later use in interpolation functions.
    
    Parameters
    ----------
    x : array_like, shape (nin * ngroups,)
        Flattened array of x-coordinates for all groups
    y : array_like, shape (nin * ngroups,)
        Flattened array of y-coordinates for all groups
    nin : int
        Number of groups
    ngroups : int
        Maximum number of points per group
    nnans : array_like, shape (nin,)
        Number of NaN values at the end of each group
        
    Returns
    -------
    ndarray, shape (nin * ngroups,)
        Array of computed linear slopes
    """
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
    """
    Precompute all Akima spline slopes for all groups on CPU.
    
    This function computes Akima spline slopes at all points using precomputed
    linear slopes, storing them for later use in interpolation functions.
    
    Parameters
    ----------
    x : array_like, shape (nin * ngroups,)
        Flattened array of x-coordinates for all groups
    y : array_like, shape (nin * ngroups,)
        Flattened array of y-coordinates for all groups
    linear_slopes : array_like, shape (nin * ngroups,)
        Precomputed linear slopes between consecutive points
    nin : int
        Number of groups
    ngroups : int
        Maximum number of points per group
    nnans : array_like, shape (nin,)
        Number of NaN values at the end of each group
        
    Returns
    -------
    ndarray, shape (nin * ngroups,)
        Array of computed Akima spline slopes
    """
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
    """
    Optimized Akima spline slope computation using precomputed linear slopes on CPU.
    
    This function computes Akima spline slopes more efficiently by reusing
    precomputed linear slopes instead of recalculating them.
    
    Parameters
    ----------
    x : array_like
        X-coordinates of data points
    y : array_like
        Y-coordinates of data points
    linear_slopes : array_like
        Precomputed linear slopes between consecutive points
    idx : int
        Index at which to compute the spline slope
    start : int
        Starting index of the data segment
    stop : int
        Ending index of the data segment (exclusive)
        
    Returns
    -------
    float
        Akima spline slope at the specified point
    """
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
    """
    Optimized CPU kernel for Akima spline interpolation with precomputed slopes.
    
    This function performs fast parallel Akima spline interpolation using
    precomputed linear and spline slopes. Uses binary search for efficient
    interval location and numba parallel execution.
    
    Parameters
    ----------
    x_new : array_like, shape (n_f,)
        New x-coordinates at which to interpolate
    x : array_like, shape (nin * ngroups,)
        Flattened array of x-coordinates for all groups
    y : array_like, shape (nin * ngroups,)
        Flattened array of y-coordinates for all groups
    linear_slopes : array_like, shape (nin * ngroups,)
        Precomputed linear slopes between consecutive points
    spline_slopes : array_like, shape (nin * ngroups,)
        Precomputed Akima spline slopes at all points
    nin : int
        Number of groups to interpolate
    ngroups : int
        Maximum number of points per group
    nnans : array_like, shape (nin,)
        Number of NaN values at the end of each group
    result : array_like, shape (nin * n_f,)
        Output array for interpolated values
    """
    
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


@numba.jit(nopython=True, parallel=True)
def akima_linear_kernel_cpu(x_new, x, y, linear_slopes,nin, ngroups, nnans, result):
    """
    CPU kernel for linear interpolation with precomputed slopes.
    
    This function performs fast parallel linear interpolation using precomputed
    slopes and binary search for interval location.
    
    Parameters
    ----------
    x_new : array_like, shape (n_f,)
        New x-coordinates at which to interpolate
    x : array_like, shape (nin * ngroups,)
        Flattened array of x-coordinates for all groups
    y : array_like, shape (nin * ngroups,)
        Flattened array of y-coordinates for all groups
    linear_slopes : array_like, shape (nin * ngroups,)
        Precomputed linear slopes between consecutive points
    nin : int
        Number of groups to interpolate
    ngroups : int
        Maximum number of points per group
    nnans : array_like, shape (nin,)
        Number of NaN values at the end of each group
    result : array_like, shape (nin * n_f,)
        Output array for interpolated values
    """
    for j in range(nin):
        start = j * ngroups
        stop = (j + 1) * ngroups - nnans[j]
        
        for i in range(x_new.shape[0]):
            special_index = j * x_new.shape[0] + i
            
            # Handle edge case
            if x_new[i] >= x[stop - 1]:
                result[special_index] = y[stop - 1]
                continue
            
            # Use binary search for interval finding
            idx = binary_search_cpu(x, x_new[i], start, stop)
            
            # Linear interpolation
            mi = linear_slopes[idx]
            result[special_index] = y[idx] + mi * (x_new[i] - x[idx])


## Multidimensional kernels

@cuda.jit
def akima_linear_kernel_gpu_multidim(x_new, x, y, linear_slopes,
                                     nin, ngroups, nnans, result):
    """
    GPU kernel for linear interpolation with multidimensional x_new arrays.
    
    This kernel handles cases where each group has its own set of interpolation
    points (x_new is 2D with shape (nin, n_f)).
    
    Parameters
    ----------
    x_new : array_like, shape (nin, n_f)
        New x-coordinates at which to interpolate, different for each group
    x : array_like, shape (nin * ngroups,)
        Flattened array of x-coordinates for all groups
    y : array_like, shape (nin * ngroups,)
        Flattened array of y-coordinates for all groups
    linear_slopes : array_like, shape (nin * ngroups,)
        Precomputed linear slopes between consecutive points
    nin : int
        Number of groups to interpolate
    ngroups : int
        Maximum number of points per group
    nnans : array_like, shape (nin,)
        Number of NaN values at the end of each group
    result : array_like, shape (nin * n_f,)
        Output array for interpolated values
    """
    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x  # x_new index within group
    j = cuda.blockIdx.y  # group index
    
    if j >= nin:
        return
    
    start = j * ngroups
    stop = (j + 1) * ngroups - nnans[j]
    
    # x_new is now multidimensional: x_new[j, i] instead of x_new[i]
    x_new_group_size = x_new.shape[1]  # assuming x_new.shape = (nin, n_f)
    
    if i >= x_new_group_size:
        return
    
    special_index = j * x_new_group_size + i
    x_target = x_new[j, i]
    
    # Handle edge case
    if x_target >= x[stop - 1]:
        result[special_index] = y[stop - 1]
        return
    
    # Use binary search for interval finding
    idx = binary_search_gpu(x, x_target, start, stop)
    
    
    # Linear interpolation
    mi = linear_slopes[idx]
    result[special_index] = y[idx] + mi * (x_target - x[idx])

@numba.jit(nopython=True, parallel=True)
def akima_linear_kernel_cpu_multidim(x_new, x, y, linear_slopes,
                                     nin, ngroups, nnans, result):
    """
    CPU kernel for linear interpolation with multidimensional x_new arrays.
    
    This function handles cases where each group has its own set of interpolation
    points (x_new is 2D with shape (nin, n_f)).
    
    Parameters
    ----------
    x_new : array_like, shape (nin, n_f)
        New x-coordinates at which to interpolate, different for each group
    x : array_like, shape (nin * ngroups,)
        Flattened array of x-coordinates for all groups
    y : array_like, shape (nin * ngroups,)
        Flattened array of y-coordinates for all groups
    linear_slopes : array_like, shape (nin * ngroups,)
        Precomputed linear slopes between consecutive points
    nin : int
        Number of groups to interpolate
    ngroups : int
        Maximum number of points per group
    nnans : array_like, shape (nin,)
        Number of NaN values at the end of each group
    result : array_like, shape (nin * n_f,)
        Output array for interpolated values
    """
    
    # Parallel loop over groups
    for j in numba.prange(nin):
        start = j * ngroups
        stop = (j + 1) * ngroups - nnans[j]
        
        x_new_group_size = x_new.shape[1]  # x_new.shape = (nin, n_f)
        
        # Sequential loop over interpolation points for this group
        for i in range(x_new_group_size):
            special_index = j * x_new_group_size + i
            x_target = x_new[j, i]
            
            # Handle edge case
            if x_target >= x[stop - 1]:
                result[special_index] = y[stop - 1]
                continue
            
            # Use binary search for interval finding
            idx = binary_search_cpu(x, x_target, start, stop)
            # Linear interpolation
            mi = linear_slopes[idx]
            result[special_index] = y[idx] + mi * (x_target - x[idx])


@cuda.jit
def akima_spline_kernel_gpu_multidim(x_new, x, y, linear_slopes, spline_slopes, 
                                     nin, ngroups, nnans, result):
    """
    GPU kernel for Akima spline interpolation with multidimensional x_new arrays.
    
    This kernel handles cases where each group has its own set of interpolation
    points (x_new is 2D with shape (nin, n_f)).
    
    Parameters
    ----------
    x_new : array_like, shape (nin, n_f)
        New x-coordinates at which to interpolate, different for each group
    x : array_like, shape (nin * ngroups,)
        Flattened array of x-coordinates for all groups
    y : array_like, shape (nin * ngroups,)
        Flattened array of y-coordinates for all groups
    linear_slopes : array_like, shape (nin * ngroups,)
        Precomputed linear slopes between consecutive points
    spline_slopes : array_like, shape (nin * ngroups,)
        Precomputed Akima spline slopes at all points
    nin : int
        Number of groups to interpolate
    ngroups : int
        Maximum number of points per group
    nnans : array_like, shape (nin,)
        Number of NaN values at the end of each group
    result : array_like, shape (nin * n_f,)
        Output array for interpolated values
    """
    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x  # x_new index within group
    j = cuda.blockIdx.y  # group index
    
    if j >= nin:
        return
    
    start = j * ngroups
    stop = (j + 1) * ngroups - nnans[j]
    
    # x_new is now multidimensional: x_new[j, i] instead of x_new[i]
    x_new_group_size = x_new.shape[1]  # assuming x_new.shape = (nin, n_f)
    
    if i >= x_new_group_size:
        return
    
    special_index = j * x_new_group_size + i
    x_target = x_new[j, i]
    
    # Handle edge case
    if x_target >= x[stop - 1]:
        result[special_index] = y[stop - 1]
        return
    
    # Use binary search for interval finding
    idx = binary_search_gpu(x, x_target, start, stop)
    
    if stop - start < 4:
        # Linear interpolation
        mi = linear_slopes[idx]
        result[special_index] = y[idx] + mi * (x_target - x[idx])
    else:
        # Akima spline interpolation
        mi = linear_slopes[idx]
        si = spline_slopes[idx]
        si1 = spline_slopes[idx + 1]
        
        dx = x[idx + 1] - x[idx]
        dt = x_target - x[idx]
        
        p0 = y[idx]
        p1 = si
        p2 = (3 * mi - 2 * si - si1) / dx
        p3 = (si + si1 - 2 * mi) / (dx * dx)
        
        result[special_index] = p0 + p1 * dt + p2 * dt * dt + p3 * dt * dt * dt

@numba.jit(nopython=True, parallel=True)
def akima_spline_kernel_cpu_multidim(x_new, x, y, linear_slopes, spline_slopes, 
                                     nin, ngroups, nnans, result):
    """
    CPU kernel for Akima spline interpolation with multidimensional x_new arrays.
    
    This function handles cases where each group has its own set of interpolation
    points (x_new is 2D with shape (nin, n_f)).
    
    Parameters
    ----------
    x_new : array_like, shape (nin, n_f)
        New x-coordinates at which to interpolate, different for each group
    x : array_like, shape (nin * ngroups,)
        Flattened array of x-coordinates for all groups
    y : array_like, shape (nin * ngroups,)
        Flattened array of y-coordinates for all groups
    linear_slopes : array_like, shape (nin * ngroups,)
        Precomputed linear slopes between consecutive points
    spline_slopes : array_like, shape (nin * ngroups,)
        Precomputed Akima spline slopes at all points
    nin : int
        Number of groups to interpolate
    ngroups : int
        Maximum number of points per group
    nnans : array_like, shape (nin,)
        Number of NaN values at the end of each group
    result : array_like, shape (nin * n_f,)
        Output array for interpolated values
    """
    
    # Parallel loop over groups
    for j in numba.prange(nin):
        start = j * ngroups
        stop = (j + 1) * ngroups - nnans[j]
        
        x_new_group_size = x_new.shape[1]  # x_new.shape = (nin, n_f)
        
        # Sequential loop over interpolation points for this group
        for i in range(x_new_group_size):
            special_index = j * x_new_group_size + i
            x_target = x_new[j, i]
            
            # Handle edge case
            if x_target >= x[stop - 1]:
                result[special_index] = y[stop - 1]
                continue
            
            # Use binary search for interval finding
            idx = binary_search_cpu(x, x_target, start, stop)
            
            if stop - start < 4:
                # Linear interpolation
                mi = linear_slopes[idx]
                result[special_index] = y[idx] + mi * (x_target - x[idx])
            else:
                # Akima spline interpolation
                mi = linear_slopes[idx]
                si = spline_slopes[idx]
                si1 = spline_slopes[idx + 1]
                
                dx = x[idx + 1] - x[idx]
                dt = x_target - x[idx]
                
                p0 = y[idx]
                p1 = si
                p2 = (3 * mi - 2 * si - si1) / dx
                p3 = (si + si1 - 2 * mi) / (dx * dx)
                
                result[special_index] = p0 + p1 * dt + p2 * dt * dt + p3 * dt * dt * dt
