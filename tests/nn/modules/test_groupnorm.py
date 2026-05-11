"""Tests for complex GroupNorm."""

from __future__ import annotations

import pytest
import torch

from complextorch.nn.modules.groupnorm import GroupNorm


def test_groupnorm_forward():
    gn = GroupNorm(num_groups=2, num_channels=8)
    x = torch.randn(4, 8, 6, 6, dtype=torch.cfloat)
    out = gn(x)
    assert out.shape == x.shape
    assert out.is_complex()


def test_groupnorm_no_affine():
    gn = GroupNorm(num_groups=2, num_channels=8, affine=False)
    x = torch.randn(4, 8, 6, 6, dtype=torch.cfloat)
    out = gn(x)
    assert gn.weight is None
    assert gn.bias is None
    assert out.shape == x.shape


def test_groupnorm_indivisible_raises():
    with pytest.raises(ValueError, match="must be divisible"):
        GroupNorm(num_groups=3, num_channels=8)


def test_groupnorm_real_input_raises():
    gn = GroupNorm(2, 4)
    with pytest.raises(TypeError, match="expects a complex input"):
        gn(torch.randn(2, 4, 4, 4))


def test_groupnorm_wrong_channels_raises():
    gn = GroupNorm(2, 4)
    with pytest.raises(ValueError, match="Expected 4 channels"):
        gn(torch.randn(2, 8, 4, 4, dtype=torch.cfloat))


def test_groupnorm_extra_repr():
    s = GroupNorm(2, 8).extra_repr()
    assert "num_groups=2" in s
    assert "num_channels=8" in s


def test_groupnorm_reset_parameters_no_op_no_affine():
    gn = GroupNorm(2, 4, affine=False)
    gn.reset_parameters()


def test_groupnorm_1d_spatial():
    """GroupNorm with no spatial dims (B, C) works too."""
    gn = GroupNorm(num_groups=2, num_channels=8)
    x = torch.randn(4, 8, 6, dtype=torch.cfloat)
    out = gn(x)
    assert out.shape == x.shape
