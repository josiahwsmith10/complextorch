"""Tests for EfficientChannelAttention{1,2,3}d."""

from __future__ import annotations

import pytest
import torch

from complextorch.nn.modules.attention.eca import (
    EfficientChannelAttention1d,
    EfficientChannelAttention2d,
    EfficientChannelAttention3d,
)


@pytest.mark.parametrize(
    ("cls", "shape"),
    [
        (EfficientChannelAttention1d, (1, 8, 16)),
        (EfficientChannelAttention2d, (1, 8, 8, 8)),
        (EfficientChannelAttention3d, (1, 8, 4, 4, 4)),
    ],
)
def test_eca_forward(cls, shape):
    eca = cls(channels=8)
    x = torch.randn(*shape, dtype=torch.cfloat)
    out = eca(x)
    assert out.shape == x.shape
    assert out.is_complex()
