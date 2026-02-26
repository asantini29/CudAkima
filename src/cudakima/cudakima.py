# -*- coding: utf-8 -*-
from importlib.metadata import version as _version

if __name__ != "__main__":
    __name__                = "cudakima"
    __version__             = _version("cudakima")
    __author__              = "Alessandro Santini"
    __author_email__        = "alessandro.santini@aei.mpg.de"
    __description__         = "GPU-accelerated, parallel implementation of Akima Splines"
    __license__             = "MIT"