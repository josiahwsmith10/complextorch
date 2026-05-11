"""Tests for complex Linear and Bilinear."""

from __future__ import annotations

import torch

from complextorch.nn.modules.linear import Bilinear, Linear


def test_linear_forward_shape_and_dtype():
    layer = Linear(8, 4, bias=True)
    x = torch.randn(3, 8, dtype=torch.cfloat)
    out = layer(x)
    assert out.shape == (3, 4)
    assert out.is_complex()


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
