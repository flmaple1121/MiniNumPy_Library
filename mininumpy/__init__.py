# mininumpy/__init__.py

"""!
@brief MiniNumPy package initializer.

This file makes the mininumpy directory a Python package and exposes key
classes and functions for easier access.
"""

from .array import Array, array, zeros, ones, arange, linspace, eye
from .array import exp, log, sqrt, abs, sum, mean, min, max, argmin, argmax
from . import linalg

__version__ = "0.1.0"