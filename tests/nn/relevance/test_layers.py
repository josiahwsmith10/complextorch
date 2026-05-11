"""Tests for the VD/ARD complex layers."""

from __future__ import annotations

import pytest
import torch

from complextorch.nn.relevance import (
    BilinearARD,
    BilinearVD,
    Conv1dARD,
    Conv1dVD,
    Conv2dARD,
    Conv2dVD,
    Conv3dARD,
    Conv3dVD,
    LinearARD,
    LinearVD,
)


# ---------- LinearVD/ARD ----------


@pytest.mark.parametrize("cls", [LinearVD, LinearARD])
@pytest.mark.parametrize("training", [True, False])
def test_linear_vd_ard_forward(cls, training):
    layer = cls(4, 6, bias=True)
    layer.train(training)
    x = torch.randn(2, 4, dtype=torch.cfloat)
    out = layer(x)
    assert out.shape == (2, 6)
    assert out.is_complex()


def test_linear_vd_penalty_finite():
    layer = LinearVD(4, 6)
    p = layer.penalty
    assert torch.isfinite(p).all()
    assert (p >= -1e-3).all()  # KL is non-negative up to floating noise


def test_linear_ard_penalty_finite():
    layer = LinearARD(4, 6)
    p = layer.penalty
    assert torch.isfinite(p).all()


def test_linear_vd_relevance_all_dropped():
    layer = LinearVD(4, 6)
    mask = layer.relevance(threshold=-1e6)  # nothing passes
    assert mask.sum().item() == 0


def test_linear_vd_relevance_all_kept():
    layer = LinearVD(4, 6)
    mask = layer.relevance(threshold=1e6)  # everything passes
    assert mask.sum().item() == mask.numel()


def test_linear_vd_sparsity_call():
    layer = LinearVD(4, 6)
    pairs = layer.sparsity(threshold=0.0)
    assert len(pairs) == 1


def test_linear_vd_no_bias():
    layer = LinearVD(4, 6, bias=False)
    assert layer.bias is None
    out = layer(torch.randn(2, 4, dtype=torch.cfloat))
    assert out.is_complex()


# ---------- BilinearVD/ARD ----------


@pytest.mark.parametrize("cls", [BilinearVD, BilinearARD])
@pytest.mark.parametrize("training", [True, False])
def test_bilinear_vd_ard_forward(cls, training):
    layer = cls(4, 5, 6, bias=True)
    layer.train(training)
    x1 = torch.randn(2, 4, dtype=torch.cfloat)
    x2 = torch.randn(2, 5, dtype=torch.cfloat)
    out = layer(x1, x2)
    assert out.shape == (2, 6)


def test_bilinear_vd_no_bias():
    layer = BilinearVD(4, 5, 6, bias=False)
    assert layer.bias is None


def test_bilinear_vd_no_conjugate():
    layer = BilinearVD(4, 5, 6, conjugate=False)
    x1 = torch.randn(2, 4, dtype=torch.cfloat)
    x2 = torch.randn(2, 5, dtype=torch.cfloat)
    out = layer(x1, x2)
    assert out.shape == (2, 6)


# ---------- ConvVD/ARD ----------


@pytest.mark.parametrize(
    "cls, shape",
    [
        (Conv1dVD, (1, 2, 8)),
        (Conv2dVD, (1, 2, 6, 6)),
        (Conv3dVD, (1, 2, 4, 4, 4)),
        (Conv1dARD, (1, 2, 8)),
        (Conv2dARD, (1, 2, 6, 6)),
        (Conv3dARD, (1, 2, 4, 4, 4)),
    ],
)
@pytest.mark.parametrize("training", [True, False])
def test_conv_vd_ard_forward(cls, shape, training):
    layer = cls(2, 4, kernel_size=3, padding=1)
    layer.train(training)
    x = torch.randn(*shape, dtype=torch.cfloat)
    out = layer(x)
    assert out.is_complex()


def test_conv_vd_no_bias():
    layer = Conv1dVD(2, 4, kernel_size=3, bias=False)
    assert layer.bias is None


def test_conv_vd_invalid_padding_mode():
    with pytest.raises(ValueError, match="padding_mode"):
        Conv1dVD(2, 4, kernel_size=3, padding_mode="reflect")


def test_conv_vd_tuple_kernel_size():
    layer = Conv2dVD(2, 4, kernel_size=(3, 5), padding=(1, 2))
    x = torch.randn(1, 2, 8, 10, dtype=torch.cfloat)
    out = layer(x)
    assert out.is_complex()


def test_conv_vd_str_padding():
    layer = Conv2dVD(2, 4, kernel_size=3, padding="same")
    x = torch.randn(1, 2, 8, 8, dtype=torch.cfloat)
    out = layer(x)
    assert out.shape == (1, 4, 8, 8)
