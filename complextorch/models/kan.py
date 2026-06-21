r"""
Complex-Valued Kolmogorov-Arnold Network (CVKAN)
================================================

A KAN-style stack of :class:`complextorch.nn.CVKANLayer` edge-function layers.
See https://arxiv.org/abs/2502.02417.
"""

import itertools

import torch
import torch.nn as nn

from complextorch.nn.modules.kan import CVKANLayer

__all__ = ["CVKAN"]


class CVKAN(nn.Module):
    r"""
    Complex-Valued Kolmogorov-Arnold Network.

    Stacks :class:`complextorch.nn.CVKANLayer` layers according to
    ``layer_sizes`` (e.g. ``[in, hidden, out]``). Operates on complex tensors of
    shape ``(..., layer_sizes[0])`` and returns ``(..., layer_sizes[-1])``.

    Args:
        layer_sizes: feature sizes from input to output (length >= 2).
        num_grid: grid resolution per axis for each :class:`CVKANLayer`.
    """

    def __init__(self, layer_sizes: list[int], num_grid: int = 8) -> None:
        super().__init__()
        if len(layer_sizes) < 2:
            raise ValueError("layer_sizes must have at least two entries (in, out)")
        self.layers = nn.ModuleList(
            [
                CVKANLayer(a, b, num_grid=num_grid)
                for a, b in itertools.pairwise(layer_sizes)
            ]
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            input = layer(input)
        return input
