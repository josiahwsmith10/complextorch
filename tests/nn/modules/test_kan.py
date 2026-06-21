"""Tests for the Complex-Valued KAN layer."""

from __future__ import annotations

import torch
import torch.nn as nn

from complextorch.nn.modules.kan import CVKANLayer


def test_cvkan_layer_shape_and_dtype():
    layer = CVKANLayer(4, 3, num_grid=6)
    x = torch.randn(5, 4, dtype=torch.cfloat)
    out = layer(x)
    assert out.shape == (5, 3)
    assert out.is_complex()


def test_cvkan_num_centers_is_grid_squared():
    layer = CVKANLayer(2, 2, num_grid=7)
    assert layer.num_centers == 49


def test_cvkan_grad_flows():
    layer = CVKANLayer(4, 3, num_grid=5)
    x = torch.randn(5, 4, dtype=torch.cfloat)
    layer(x).abs().sum().backward()
    assert layer.spline_weight.grad is not None
    assert torch.isfinite(layer.spline_weight.grad).all()


def test_cvkan_learnable_grid():
    layer = CVKANLayer(3, 2, num_grid=4, learnable_grid=True)
    assert isinstance(layer.centers, nn.Parameter)
    x = torch.randn(6, 3, dtype=torch.cfloat)
    layer(x).abs().sum().backward()
    assert layer.centers.grad is not None


def test_cvkan_fixed_grid_is_buffer():
    layer = CVKANLayer(3, 2, num_grid=4)
    assert not isinstance(layer.centers, nn.Parameter)


def test_cvkan_can_fit_a_complex_function():
    """A few optimisation steps reduce the fit error on z -> z**2."""
    torch.manual_seed(0)
    layer = CVKANLayer(1, 1, num_grid=8)
    x = (torch.rand(256, 1) * 1.6 - 0.8) + 1j * (torch.rand(256, 1) * 1.6 - 0.8)
    y = x**2
    opt = torch.optim.Adam(layer.parameters(), lr=0.05)
    init_loss = (layer(x) - y).abs().pow(2).mean().item()
    for _ in range(100):
        opt.zero_grad()
        loss = (layer(x) - y).abs().pow(2).mean()
        loss.backward()
        opt.step()
    assert loss.item() < init_loss * 0.5


def test_cvkan_extra_repr():
    s = CVKANLayer(4, 3, num_grid=6).extra_repr()
    assert "in_features=4" in s
    assert "num_grid=6" in s
