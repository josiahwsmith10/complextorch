"""Tests for wFM modules: convolutions, ReLU, distance linear."""

from __future__ import annotations

import torch

from complextorch.nn.modules.manifold import (
    wFMConv1d,
    wFMConv2d,
    wFMDistanceLinear,
    wFMReLU,
)


def test_wfm_conv2d_forward():
    conv = wFMConv2d(in_channels=3, out_channels=5, kernel_size=(3, 3), padding=(1, 1))
    x = torch.randn(2, 3, 8, 8, dtype=torch.cfloat) + 0.1
    out = conv(x)
    assert out.is_complex()
    assert out.shape[0] == 2


def test_wfm_conv2d_fold_cache_reused():
    conv = wFMConv2d(in_channels=3, out_channels=5, kernel_size=(3, 3), padding=(1, 1))
    x = torch.randn(2, 3, 8, 8, dtype=torch.cfloat) + 0.1
    conv(x)
    n_before = len(conv._fold_cache)
    conv(x)
    n_after = len(conv._fold_cache)
    assert n_before == n_after


def test_wfm_conv1d_forward():
    conv = wFMConv1d(in_channels=3, out_channels=5, kernel_size=3, padding=1)
    x = torch.randn(2, 3, 8, dtype=torch.cfloat) + 0.1
    out = conv(x)
    assert out.is_complex()


def test_wfm_conv1d_weight_properties():
    conv = wFMConv1d(in_channels=3, out_channels=5, kernel_size=3)
    assert conv.weight_matrix_ang is conv.conv1d.weight_matrix_ang
    assert conv.weight_matrix_mag is conv.conv1d.weight_matrix_mag


# ---------- wFMReLU ----------


def test_wfm_relu_forward_shape():
    layer = wFMReLU(num_channels=4)
    x = torch.randn(2, 4, 6, 6, dtype=torch.cfloat) + 0.1
    out = layer(x)
    assert out.shape == x.shape
    assert out.is_complex()


def test_wfm_relu_grad_flows():
    layer = wFMReLU(num_channels=3)
    x = torch.randn(2, 3, 4, dtype=torch.cfloat) + 0.1
    out = layer(x)
    out.abs().sum().backward()
    assert layer.weight_phase.grad is not None
    assert layer.weight_mag.grad is not None
    assert torch.isfinite(layer.weight_phase.grad).all()


def test_wfm_relu_extra_repr():
    s = wFMReLU(num_channels=4).extra_repr()
    assert "num_channels=4" in s


# ---------- wFMDistanceLinear ----------


def test_wfm_distance_linear_returns_real():
    layer = wFMDistanceLinear(input_dim=4 * 6)
    x = torch.randn(2, 4, 6, dtype=torch.cfloat) + 0.1
    out = layer(x)
    assert out.shape == x.shape  # preserves shape of complex input
    assert not out.is_complex()
    assert out.dtype == torch.float32 or out.dtype == torch.float64


def test_wfm_distance_linear_grad_flows():
    layer = wFMDistanceLinear(input_dim=12)
    x = torch.randn(2, 3, 4, dtype=torch.cfloat) + 0.1
    out = layer(x)
    out.sum().backward()
    assert layer.weights.grad is not None
    assert torch.isfinite(layer.weights.grad).all()


def test_wfm_distance_linear_extra_repr():
    s = wFMDistanceLinear(input_dim=10).extra_repr()
    assert "input_dim=10" in s
