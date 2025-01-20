# CudAkima: Parallel Akima Splines on GPUs

CudAkima is a Python package that offers a parallel, GPU-accelerated implementation of Akima Splines. The code also provides CPU support. 

## Getting started:

[Akima Splines](https://en.wikipedia.org/wiki/Akima_spline) are spline interpolants that tend to show smoother behaviors with respect to the widely used Cubic Splines. On the other hand, Akima Splines have discontinuous second derivative.

Both `scipy` and `cupy` already offer an implementation of Akima Splines. However, in these versions, the $x$ array the interpolation is performed along can only be 1D. The main point behind the whole package is the fast, parallel interpolation of batches of arrays of different lengths. This is achieved by padding the different 1D arrays with `NaN` and stacking them in multidimensional arrays.

In this implementation, the coefficients of the polynomials used for the interpolation are not saved and kept in memory. For this reason, the package is particularly suited for applications where the arrays to perform the interpolation on keep changing (e.g. when doing parameter estimation on the location and amplitude of the spline knots). In this specific case, where $x$ and $y$ are matrices of different arrays, CudAkima results faster than a naive loop over the matrices using `scipy` (`cupy`) by a factor of $\sim 3$ ($\sim 20$) on CPUs (GPUs). This comparison can be found in the examples directory.

The interpolation scheme needs at least 4 finite points to succesfully work. This caveat is due to the boundary conditions currently implemented. If this condition is not met (ie, the interpolation grid is made of less than 4 points), **linear interpolation** is used instead. 

 Here is a quick example of how to get started with the package:
```
from cudakima import AkimaInterpolant1D

interpolant = AkimaInterpolant1D()
```
Check out the examples directory for more info and comparisons.

### Prerequisites:

CudAkima depends only on `numba` and `numpy`. It also requires `cupy` to be used on GPUs.

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

Current Version: 0.0.2

## Authors

* **Alesandro Santini**

### Contributors

Get in touch if you would like to contribute!

## Code TODO:
* extend documentation.
* look at different boundary conditions.
* work on a possible 2D interpolation.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE) file for details.

## Citing

If you use CudAkima in your research, you can cite it in the following way:

```
@software{cudakima_2024_13919394,
  author       = {Alessandro Santini},
  title        = {asantini29/CudAkima: First official release},
  month        = oct,
  year         = 2024,
  publisher    = {Zenodo},
  version      = {v0.0.1},
  doi          = {10.5281/zenodo.13919394},
  url          = {https://doi.org/10.5281/zenodo.13919394}
}
```

## Aknowledgments
We thank Nikolaos Karnesis for discussions.
