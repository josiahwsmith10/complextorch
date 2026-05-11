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
    named_penalties,
    penalties,
    named_relevance,
    compute_ard_masks,
)
from complextorch.nn.relevance.linear import (
    LinearVD,
    BilinearVD,
    LinearARD,
    BilinearARD,
)
from complextorch.nn.relevance.conv import (
    Conv1dVD,
    Conv2dVD,
    Conv3dVD,
    Conv1dARD,
    Conv2dARD,
    Conv3dARD,
)

__all__ = [
    "BaseARD",
    "named_penalties",
    "penalties",
    "named_relevance",
    "compute_ard_masks",
    "LinearVD",
    "BilinearVD",
    "LinearARD",
    "BilinearARD",
    "Conv1dVD",
    "Conv2dVD",
    "Conv3dVD",
    "Conv1dARD",
    "Conv2dARD",
    "Conv3dARD",
]
