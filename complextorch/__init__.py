r"""
complextorch
------------
Author: Josiah W. Smith

A lightweight complex-valued neural network package built on PyTorch.

For more information see: https://pypi.org/project/complextorch or https://github.com/josiahwsmith10/complextorch
"""

__author__ = "Josiah W. Smith"

__version__ = "1.1.4"

__all__ = ["CVTensor", "cat", "roll", "from_polar", "nn", "randn"]

from ._cvtensor import CVTensor
from ._cvtensor import cat
from ._cvtensor import roll
from ._cvtensor import from_polar

from . import nn

from ._cvrandom import randn
