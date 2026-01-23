Examples
========

Additional Examples
-------------------

For more examples and benchmarks, check out the `examples directory <https://github.com/asantini29/CudAkima/tree/main/examples>`_ 
in the repository.

Basic Interpolation
~~~~~~~~~~~~~~~~~~~

Single Array Interpolation
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from cudakima import AkimaInterpolant1D
   import numpy as np

   # Simple single array interpolation
   x = np.array([1, 2, 3, 4, 5])
   y = np.array([1, 4, 9, 16, 25])
   
   interpolant = AkimaInterpolant1D()
   x_new = np.linspace(1, 5, 50)
   y_new = interpolant(x_new, x, y)

Batch Interpolation with Different Lengths
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from cudakima import AkimaInterpolant1D
   import numpy as np

   # Batch interpolation with NaN padding
   x = np.array([
       [1, 2, 3, 4, 5, np.nan, np.nan],
       [1, 2, 3, 4, 5, 6, np.nan],
       [1, 2, 3, 4, 5, 6, 7]
   ])
   y = np.array([
       [1, 4, 9, 16, 25, np.nan, np.nan],
       [1, 4, 9, 16, 25, 36, np.nan],
       [1, 4, 9, 16, 25, 36, 49]
   ])
   
   interpolant = AkimaInterpolant1D()
   x_new = np.linspace(1, 5, 100)
   y_new = interpolant(x_new, x, y)  # Shape: (3, 100)

Advanced Usage
~~~~~~~~~~~~~~

Linear Interpolation
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from cudakima import AkimaInterpolant1D
   
   # Use linear interpolation instead of cubic
   interpolant = AkimaInterpolant1D(order='linear')
   y_new = interpolant(x_new, x, y)

CPU-Only Mode
^^^^^^^^^^^^^

.. code-block:: python

   from cudakima import AkimaInterpolant1D
   
   # Force CPU execution (useful for debugging or when GPU is unavailable)
   interpolant = AkimaInterpolant1D(use_gpu=False)
   y_new = interpolant(x_new, x, y)

Multidimensional x_new
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from cudakima import AkimaInterpolant1DMultiDim
   import numpy as np

   # Different interpolation points for each group
   x = np.random.rand(2, 3, 10)  # 2x3 batch, 10 points each
   y = np.sin(x)
   x_new = np.random.rand(2, 3, 50)  # 2x3 batch, 50 interp points each
   
   interpolant = AkimaInterpolant1DMultiDim()
   y_new = interpolant(x_new, x, y)  # Shape: (2, 3, 50)

Flexible Interpolator
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from cudakima import AkimaInterpolant1DFlexible
   
   # Automatically chooses between standard and multidimensional modes
   interpolant = AkimaInterpolant1DFlexible()
   
   # Standard mode (1D x_new)
   y_new_standard = interpolant(x_new_1d, x, y)
   
   # Multidimensional mode (matching batch dims)
   y_new_multidim = interpolant(x_new_multidim, x, y)

Performance Tips
~~~~~~~~~~~~~~~~

1. **Presorting**: If your input data is already sorted, set ``sanitize=False`` to skip sorting:

   .. code-block:: python

      interpolant = AkimaInterpolant1D(sanitize=False)

2. **Batch Size**: For optimal GPU performance, use larger batch sizes (more groups to interpolate).

3. **Thread Configuration**: Adjust ``threadsperblock`` for your GPU (default is 64):

   .. code-block:: python

      interpolant = AkimaInterpolant1D(threadsperblock=128)

4. **GPU Memory**: For very large datasets, consider processing in chunks to avoid out-of-memory errors.
