"""Tests for the Steinmetz and Analytic neural networks."""

from __future__ import annotations

import torch

from complextorch.models.steinmetz import AnalyticNeuralNetwork, SteinmetzNetwork
from complextorch.signal import analytic_signal


def test_steinmetz_forward_shape_and_dtype():
    net = SteinmetzNetwork(in_features=4, hidden_features=8, out_features=3)
    x = torch.randn(5, 4, dtype=torch.cfloat)
    out = net(x)
    assert out.shape == (5, 3)
    assert out.is_complex()


def test_steinmetz_grad_flows():
    net = SteinmetzNetwork(4, 8, 3, depth=3)
    x = torch.randn(5, 4, dtype=torch.cfloat)
    net(x).abs().sum().backward()
    grads = [p.grad for p in net.parameters()]
    assert all(g is not None for g in grads)


def test_steinmetz_depth_one():
    # depth=1 -> a single linear layer per branch (no hidden ReLU stack).
    net = SteinmetzNetwork(4, 8, 2, depth=1)
    out = net(torch.randn(3, 4, dtype=torch.cfloat))
    assert out.shape == (3, 2)


def test_analytic_network_forward_and_consistency():
    net = AnalyticNeuralNetwork(4, 8, 16, depth=2)
    x = torch.randn(5, 4, dtype=torch.cfloat)
    out = net(x)
    assert out.shape == (5, 16)
    pen = net.consistency_loss(out)
    assert pen.dim() == 0
    assert torch.isfinite(pen).all()


def test_analytic_network_consistency_zero_on_analytic_latent():
    net = AnalyticNeuralNetwork(4, 8, 64)
    z = analytic_signal(torch.randn(3, 64))  # genuine analytic signal
    torch.testing.assert_close(
        net.consistency_loss(z), torch.zeros(()), atol=1e-10, rtol=0
    )
