# CudAkima: Parallel Akima Splines on GPUs

CudAkima is a Python package which offers a parallel, GPU-accelerated implementation of Akima Splines. The code offers CPU support as well. 

## Getting started:

[Akima Splines](https://en.wikipedia.org/wiki/Akima_spline) are spline interpolants that tend to show smoother behaviours with respect to the widely used Cubic Splines. On the other hand, Akima Splines have discontinuous second derivative.

Both `scipy` and `cupy` already offer an implementation of Akima Splines. Anyway, in these versions the $x$ array the interpolation is performed along can only be 1D. The main point behind the whole package is the fast, parallel interpolation of batches of arrays of different lenghts. This is achieved padding the different 1D arrays with `NaN` and stacking them in multidimensional arrays.

 Here is a quick example of how to get started with the package:
```
from cudakima import AkimaInterpolant1D

interpolant = AkimaInterpolant1D()
```
Check out the examples directory for more info and comparisons.

### Prerequisites:

CudAkima depends only on `numba` and `numpy`. In order to be used on GPUs, also `cupy` is required.

## Installing:
1. Clone the repository:
   ```
   git clone https://github.com/asantini29/CudAkima.git
   cd CudAkima
   ```
2. Run install:
   ```
   python setup.py install
   ```

## Versioning

We use [SemVer](http://semver.org/) for versioning. 

Current Version: 0.0.1

## Authors

* **Alesandro Santini**

### Contibutors

Get in touch if you would like to contribute!

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE) file for details.
