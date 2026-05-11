"""Tests for MaskedChannelAttention{1,2,3}d."""

from __future__ import annotations

import pytest
import torch

from complextorch.nn.modules.attention.mca import (
    MaskedChannelAttention1d,
    MaskedChannelAttention2d,
    MaskedChannelAttention3d,
)


@pytest.mark.parametrize(
    ("cls", "shape"),
    [
        (MaskedChannelAttention1d, (1, 8, 16)),
        (MaskedChannelAttention2d, (1, 8, 8, 8)),
        (MaskedChannelAttention3d, (1, 8, 4, 4, 4)),
    ],
)
def test_mca_forward(cls, shape):
    mca = cls(channels=8, reduction_factor=2)
    x = torch.randn(*shape, dtype=torch.cfloat)
    out = mca(x)
    assert out.shape == x.shape
    assert out.is_complex()


def test_mca_invalid_reduction_factor():
    with pytest.raises(AssertionError, match="yield integer"):
        MaskedChannelAttention1d(channels=8, reduction_factor=3)
