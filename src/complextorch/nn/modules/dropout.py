import torch.nn as nn

from ... import CVTensor
from .. import functional as cvF

__all__ = ['CVDropout']


class CVDropout(nn.Module):
    """Complex-Valued Dropout Layer."""

    def __init__(self, p: float = 0.5, inplace: bool = False) -> None:
        super(CVDropout, self).__init__()

        self.dropout_r = nn.Dropout(p, inplace)
        self.dropout_i = nn.Dropout(p, inplace)

    def forward(self, x: CVTensor) -> CVTensor:
        return cvF.apply_complex_split(self.dropout_r, self.dropout_i, x)
