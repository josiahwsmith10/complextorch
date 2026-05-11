"""Hand-rolled real/imag-split layers using Gauss' multiplication trick.

These variants compute complex linear / convolution with 3 real operations
instead of the naive 4, but they are typically *slower* than the native
PyTorch complex kernels on modern PyTorch (>= 2.1.0). They are kept as
reference implementations and for users who want explicit access to the
real/imag halves (e.g. for parameterization tricks). For ordinary use,
prefer ``complextorch.nn.Conv*d`` / ``complextorch.nn.Linear``, which wrap
``torch.nn.*`` with ``dtype=torch.cfloat``.
"""

from complextorch.nn.gauss.linear import Linear
from complextorch.nn.gauss.conv import (
    Conv1d,
    Conv2d,
    Conv3d,
    ConvTranspose1d,
    ConvTranspose2d,
    ConvTranspose3d,
)

__all__ = [
    "Linear",
    "Conv1d",
    "Conv2d",
    "Conv3d",
    "ConvTranspose1d",
    "ConvTranspose2d",
    "ConvTranspose3d",
]
