# Documentation Build Guide

This guide explains how to build the CudAkima documentation locally.

## Prerequisites

You need to install the documentation dependencies:

```bash
# Using pip
pip install -e ".[docs]"

# Or install dependencies individually
pip install sphinx>=7.0.0 sphinx-rtd-theme>=2.0.0 nbsphinx>=0.9.0 ipython>=8.0.0
```

## Building the Documentation

To build the HTML documentation:

```bash
cd docs
make html
```

The generated documentation will be in `docs/build/html/`. Open `docs/build/html/index.html` in your browser to view it.

## Cleaning Build Files

To clean the build directory:

```bash
cd docs
make clean
```

## Other Output Formats

Sphinx supports multiple output formats. Some useful ones:

```bash
# PDF (requires LaTeX)
make latexpdf

# Plain text
make text

# ePub
make epub
```

## GitHub Pages Deployment

The documentation is automatically built and deployed to GitHub Pages when changes are pushed to the `main` branch. The workflow is defined in `.github/workflows/docs.yml`.

To enable GitHub Pages:

1. Go to your repository settings on GitHub
2. Navigate to "Pages" in the left sidebar
3. Under "Source", select "GitHub Actions"
4. The documentation will be available at `https://<username>.github.io/<repository>/`

## Documentation Structure

```
docs/
├── source/
│   ├── conf.py           # Sphinx configuration
│   ├── index.rst         # Main documentation page
│   ├── api.rst           # API reference
│   ├── tutorial.rst      # Tutorial page
│   ├── examples.rst      # Examples page
│   ├── notebooks/        # Jupyter notebooks
│   │   └── tutorial.ipynb (symlink to examples/tutorial.ipynb)
│   ├── _static/          # Static files (CSS, images, etc.)
│   └── _templates/       # Custom templates
├── build/                # Generated documentation (git-ignored)
└── Makefile             # Build commands
```

## Updating Documentation

1. **Class/Function Documentation**: Update docstrings in the source code (`src/cudakima/`)
2. **Tutorial**: Edit `examples/tutorial.ipynb`
3. **Examples**: Edit `docs/source/examples.rst`
4. **Main Pages**: Edit `.rst` files in `docs/source/`

After making changes, rebuild the documentation to see the updates:

```bash
cd docs
make clean
make html
```

## Docstring Format

CudAkima uses NumPy-style docstrings. Here's an example:

```python
def my_function(x, y):
    """
    Short description.
    
    Longer description if needed.
    
    Parameters
    ----------
    x : array_like
        Description of x
    y : float
        Description of y
        
    Returns
    -------
    result : ndarray
        Description of return value
        
    Examples
    --------
    >>> my_function([1, 2, 3], 2.0)
    array([2., 4., 6.])
    """
    pass
```

## Troubleshooting

### Import Errors

If you get import errors when building the documentation, make sure the package is installed:

```bash
pip install -e .
```

### Notebook Execution Errors

The notebooks are set to not execute during the build (`nbsphinx_execute = 'never'`). If you want to execute them, change this setting in `docs/source/conf.py`.

### Missing Dependencies

If you get warnings about missing dependencies, install them:

```bash
pip install -e ".[docs]"
```
