"""Tests for Upsample / PolarUpsample."""

from __future__ import annotations

import torch

from complextorch.nn.modules.upsampling import PolarUpsample, Upsample


def test_upsample_split_complex():
    up = Upsample(scale_factor=2.0, mode="nearest")
    x = torch.randn(1, 2, 4, 4, dtype=torch.cfloat)
    out = up(x)
    assert out.shape == (1, 2, 8, 8)
    assert out.is_complex()


def test_upsample_split_real_input():
    up = Upsample(scale_factor=2.0, mode="nearest")
    x = torch.randn(1, 2, 4, 4)
    out = up(x)
    assert out.shape == (1, 2, 8, 8)
    assert not out.is_complex()


def test_upsample_with_size():
    up = Upsample(size=(8, 8), mode="bilinear", align_corners=False)
    x = torch.randn(1, 2, 4, 4, dtype=torch.cfloat)
    out = up(x)
    assert out.shape == (1, 2, 8, 8)


def test_upsample_extra_repr():
    s = Upsample(scale_factor=2.0).extra_repr()
    assert "scale_factor" in s


def test_polar_upsample_complex():
    up = PolarUpsample(scale_factor=2.0, mode="nearest")
    x = torch.randn(1, 2, 4, 4, dtype=torch.cfloat)
    out = up(x)
    assert out.shape == (1, 2, 8, 8)
    assert out.is_complex()


def test_polar_upsample_real_input():
    up = PolarUpsample(scale_factor=2.0, mode="nearest")
    x = torch.randn(1, 2, 4, 4)
    out = up(x)
    assert not out.is_complex()


def test_polar_upsample_extra_repr():
    s = PolarUpsample(scale_factor=2.0).extra_repr()
    assert "scale_factor" in s
