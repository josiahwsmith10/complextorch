from typing import Union, Tuple

import torch.nn as nn

from ... import CVTensor
from .. import functional as cvF

__all__ = ["CVAdaptiveAvgPool1d", "CVAdaptiveAvgPool2d", "CVAdaptiveAvgPool3d"]


class CVAdaptiveAvgPool1d(nn.AdaptiveAvgPool1d):
    def __init__(self, output_size: Union[int, Tuple[int]]) -> None:
        super().__init__(output_size)

    def forward(self, input: CVTensor) -> CVTensor:
        return cvF.apply_complex_split(super().forward, super().forward, input)


class CVAdaptiveAvgPool2d(nn.AdaptiveAvgPool2d):
    def __init__(self, output_size) -> None:
        super().__init__(output_size)

    def forward(self, input: CVTensor) -> CVTensor:
        return cvF.apply_complex_split(super().forward, super().forward, input)


class CVAdaptiveAvgPool3d(nn.AdaptiveAvgPool3d):
    def __init__(self, output_size) -> None:
        super().__init__(output_size)

    def forward(self, input: CVTensor) -> CVTensor:
        return cvF.apply_complex_split(super().forward, super().forward, input)
