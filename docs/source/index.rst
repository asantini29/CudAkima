.. CudAkima documentation master file

Welcome to CudAkima's Documentation
===================================

CudAkima is a Python package that provides a parallel, GPU-accelerated implementation 
of Akima Splines. The package also includes CPU support for systems without CUDA/CuPy.

**Key Features:**

* GPU-accelerated parallel Akima spline interpolation
* CPU fallback for systems without GPU support
* Efficient batch interpolation of arrays with different lengths
* Support for both linear and cubic (Akima spline) interpolation
* Multidimensional interpolation support

Quick Start
-----------

Installation
~~~~~~~~~~~~

CudAkima requires ``numpy`` and ``numba``. For GPU support, you'll also need ``cupy``.

.. code-block:: bash

   git clone https://github.com/asantini29/CudAkima.git
   cd CudAkima
   uv sync

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   from cudakima import AkimaInterpolant1D
   import numpy as np

   # Create sample data
   x = np.array([[1, 2, 3, 4, 5, np.nan, np.nan], 
                 [1, 2, 3, 4, 5, 6, np.nan], 
                 [1, 2, 3, 4, 5, 6, 7]])
   y = np.array([[1, 4, 9, 16, 25, np.nan, np.nan], 
                 [1, 4, 9, 16, 25, 36, np.nan], 
                 [1, 4, 9, 16, 25, 36, 49]])

   # Create interpolator
   interpolant = AkimaInterpolant1D()

   # Interpolate
   x_new = np.linspace(1, 5, 100)
   y_new = interpolant(x_new, x, y)

About Akima Splines
-------------------

`Akima Splines <https://en.wikipedia.org/wiki/Akima_spline>`_ are spline interpolants 
that tend to show smoother behavior with respect to the widely used Cubic Splines. 
Unlike cubic splines, Akima splines have discontinuous second derivatives, which can 
be advantageous in certain applications.

Why CudAkima?
~~~~~~~~~~~~~

While both ``scipy`` and ``cupy`` offer implementations of Akima splines, they only 
support 1D x-arrays for interpolation. **CudAkima** enables fast, parallel interpolation 
of batches of arrays with different lengths by padding shorter arrays with NaN values 
and stacking them in multidimensional arrays.

This makes CudAkima particularly suited for applications where:

* The arrays to interpolate keep changing (e.g., parameter estimation)
* You need to interpolate many arrays in parallel
* Arrays in the batch have different lengths

Performance
~~~~~~~~~~~

On CPU, CudAkima is approximately **3x faster** than a naive loop using scipy.
On GPU, CudAkima is approximately **20x faster** than using cupy in a loop.

See the tutorial notebook in the examples directory for detailed benchmarks.

Table of Contents
-----------------

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   api
   tutorial
   examples

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
