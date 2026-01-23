API Reference
=============

This page documents the complete API for CudAkima.

Main Classes
------------

.. currentmodule:: cudakima

AkimaInterpolant1D
~~~~~~~~~~~~~~~~~~

.. autoclass:: cudakima.AkimaInterpolant1D
   :members:
   :special-members: __init__, __call__
   :show-inheritance:

AkimaInterpolant1DMultiDim
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: cudakima.AkimaInterpolant1DMultiDim
   :members:
   :special-members: __init__, __call__
   :show-inheritance:

AkimaInterpolant1DFlexible
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: cudakima.AkimaInterpolant1DFlexible
   :members:
   :special-members: __init__, __call__
   :show-inheritance:

Kernel Functions
----------------

This section documents the low-level numba kernels used for GPU and CPU computation.

GPU Kernels
~~~~~~~~~~~

.. currentmodule:: cudakima.kernels

.. autofunction:: linearslope_gpu

.. autofunction:: splineslope_gpu

.. autofunction:: splineslope_gpu_optimized

.. autofunction:: binary_search_gpu

.. autofunction:: akima_spline_kernel_gpu

.. autofunction:: akima_spline_kernel_optimized

.. autofunction:: akima_linear_kernel

.. autofunction:: precompute_slopes_kernel

.. autofunction:: precompute_spline_slopes_kernel

.. autofunction:: akima_linear_kernel_gpu_multidim

.. autofunction:: akima_spline_kernel_gpu_multidim

CPU Kernels
~~~~~~~~~~~

.. autofunction:: linearslope_cpu

.. autofunction:: splineslope_cpu

.. autofunction:: splineslope_cpu_optimized

.. autofunction:: binary_search_cpu

.. autofunction:: akima_spline_kernel_cpu

.. autofunction:: akima_spline_kernel_cpu_optimized

.. autofunction:: akima_linear_kernel_cpu

.. autofunction:: precompute_all_linear_slopes_cpu

.. autofunction:: precompute_all_spline_slopes_cpu

.. autofunction:: akima_linear_kernel_cpu_multidim

.. autofunction:: akima_spline_kernel_cpu_multidim
