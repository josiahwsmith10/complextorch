"""Tests for complex Linear, SlowLinear, and Bilinear."""

from __future__ import annotations

import torch

from complextorch.nn.modules.linear import Bilinear, Linear, SlowLinear


def test_linear_forward_shape_and_dtype():
    layer = Linear(8, 4, bias=True)
    x = torch.randn(3, 8, dtype=torch.cfloat)
    out = layer(x)
    assert out.shape == (3, 4)
    assert out.is_complex()


def test_slow_linear_forward():
    layer = SlowLinear(8, 4, bias=True)
    x = torch.randn(3, 8, dtype=torch.cfloat)
    out = layer(x)
    assert out.shape == (3, 4)
    assert out.is_complex()
    # weight property
    w = layer.weight
    assert w.shape == (4, 8)
    assert w.is_complex()
    # bias property
    b = layer.bias
    assert b.shape == (4,)


def test_slow_linear_no_bias():
    layer = SlowLinear(8, 4, bias=False)
    x = torch.randn(3, 8, dtype=torch.cfloat)
    out = layer(x)
    assert out.shape == (3, 4)
    assert layer.bias is None


def test_bilinear_hermitian_default():
    layer = Bilinear(
        in1_features=4, in2_features=6, out_features=3, bias=True, conjugate=True
    )
    x1 = torch.randn(2, 4, dtype=torch.cfloat)
    x2 = torch.randn(2, 6, dtype=torch.cfloat)
    out = layer(x1, x2)
    assert out.shape == (2, 3)
    assert out.is_complex()
    # Hermitian form: when x1 is real, conjugate is the identity
    x1r = torch.randn(2, 4, dtype=torch.float).to(torch.cfloat)
    out_real = layer(x1r, x2)
    expected = torch.einsum("...i,kij,...j->...k", x1r, layer.weight, x2) + layer.bias
    torch.testing.assert_close(out_real, expected)


def test_bilinear_plain_no_conjugate():
    layer = Bilinear(4, 6, 3, conjugate=False, bias=True)
    x1 = torch.randn(2, 4, dtype=torch.cfloat)
    x2 = torch.randn(2, 6, dtype=torch.cfloat)
    out = layer(x1, x2)
    expected = torch.einsum("...i,kij,...j->...k", x1, layer.weight, x2) + layer.bias
    torch.testing.assert_close(out, expected)


def test_bilinear_no_bias():
    layer = Bilinear(4, 6, 3, bias=False)
    x1 = torch.randn(2, 4, dtype=torch.cfloat)
    x2 = torch.randn(2, 6, dtype=torch.cfloat)
    out = layer(x1, x2)
    assert layer.bias is None
    expected = torch.einsum("...i,kij,...j->...k", x1.conj(), layer.weight, x2)
    torch.testing.assert_close(out, expected)


def test_bilinear_extra_repr():
    repr_str = Bilinear(4, 6, 3, conjugate=False).extra_repr()
    assert "in1_features=4" in repr_str
    assert "in2_features=6" in repr_str
    assert "out_features=3" in repr_str
    assert "conjugate=False" in repr_str


def test_linear_vs_slow_linear_numerical_equivalence():
    """Slow and Fast should agree when state dicts are aligned."""
    in_f, out_f = 6, 8
    fast = Linear(in_f, out_f, bias=True)
    slow = SlowLinear(in_f, out_f, bias=True)
    # Manually copy fast weights into slow's real/imag form
    with torch.no_grad():
        slow.linear_r.weight.copy_(fast.linear.weight.real)
        slow.linear_i.weight.copy_(fast.linear.weight.imag)
        slow.bias_r.copy_(fast.linear.bias.real)
        slow.bias_i.copy_(fast.linear.bias.imag)
    x = torch.randn(4, in_f, dtype=torch.cfloat)
    torch.testing.assert_close(fast(x), slow(x), rtol=1e-5, atol=1e-5)
