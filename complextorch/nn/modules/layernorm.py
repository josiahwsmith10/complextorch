from typing import Union, List

import torch
import torch.nn as nn
from torch.nn import init

from .. import functional as cvF
from ... import CVTensor

__all__ = ["CVLayerNorm"]


class CVLayerNorm(nn.Module):
    r"""
    Complex-Valued Layer Normalization
    ----------------------------------

    Uses whitening transformation to ensure standard normal complex distribution
    with equal variance in both real and imaginary components.

    Extending the batch normalization whitening definitions in the following paper:

        **J. A. Barrachina, C. Ren, G. Vieillard, C. Morisseau, and J.-P. Ovarlez. Theory and Implementation of Complex-Valued Neural Networks.**

            - Section 6

            - https://arxiv.org/abs/2302.08286
    """

    def __init__(
        self,
        normalized_shape: Union[int, List[int], torch.Size],
        *,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
    ) -> None:
        super().__init__()

        # Convert `normalized_shape` to `torch.Size`
        if isinstance(normalized_shape, int):
            normalized_shape = torch.Size([normalized_shape])
        elif isinstance(normalized_shape, list):
            normalized_shape = torch.Size(normalized_shape)
        assert isinstance(normalized_shape, torch.Size)

        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        # Create parameters for gamma and beta for weight and bias
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(2, 2, *normalized_shape))
            self.bias = nn.Parameter(torch.zeros(2, *normalized_shape))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if not self.elementwise_affine:
            return

        self.weight.data.copy_(
            0.70710678118 * torch.eye(2).view(2, 2, *([1] * len(self.normalized_shape)))
        )
        init.zeros_(self.bias)

    def forward(self, input: CVTensor) -> CVTensor:
        # Sanity check to make sure the shapes match
        assert (
            self.normalized_shape == input.shape[-len(self.normalized_shape) :]
        ), "Expected normalized_shape to match last dimensions of input shape!"

        return cvF.cv_layer_norm(
            input, self.normalized_shape, self.weight, self.bias, self.eps
        )
