r"""
Variational Dropout / Automatic Relevance Determination (Complex)
=================================================================

Complex-valued layers that learn a per-weight relevance score during training
via the local reparameterization trick. Adds a per-module
:attr:`BaseARD.penalty` (KL divergence) that the user adds to the negative
log-likelihood, and a :meth:`BaseARD.relevance` method that returns the
post-training binary keep/drop mask.

Adapted from :mod:`cplxmodule.nn.relevance.complex` for native ``torch.cfloat``.
"""

from complextorch.nn.relevance.base import (
    BaseARD,
    compute_ard_masks,
    named_penalties,
    named_relevance,
    penalties,
)
from complextorch.nn.relevance.conv import (
    Conv1dARD,
    Conv1dVD,
    Conv2dARD,
    Conv2dVD,
    Conv3dARD,
    Conv3dVD,
)
from complextorch.nn.relevance.linear import (
    BilinearARD,
    BilinearVD,
    LinearARD,
    LinearVD,
)

__all__ = [
    "BaseARD",
    "BilinearARD",
    "BilinearVD",
    "Conv1dARD",
    "Conv1dVD",
    "Conv2dARD",
    "Conv2dVD",
    "Conv3dARD",
    "Conv3dVD",
    "LinearARD",
    "LinearVD",
    "compute_ard_masks",
    "named_penalties",
    "named_relevance",
    "penalties",
]
