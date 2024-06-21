# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
import logging


log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())

from .cudakima import __name__
from .cudakima import __version__
from .cudakima import __description__
from .cudakima import __author__
from .cudakima import __author_email__

from .akima import AkimaInterpolant1D