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


def test_linear_weight_bias_are_live_parameters():
    """weight/bias are the wrapped module's parameters, storage-shared."""
    layer = Linear(8, 4)
    assert layer.weight is layer.linear.weight
    assert layer.bias is layer.linear.bias
    x = torch.randn(2, 8, dtype=torch.cfloat)
    y_before = layer(x).detach().clone()
    layer.weight.data.zero_()
    assert not torch.allclose(layer(x), y_before)


def test_linear_bias_none_when_disabled():
    assert Linear(8, 4, bias=False).bias is None


def test_linear_parameter_assignment_rebinds():
    layer = Linear(8, 4)
    p = torch.nn.Parameter(torch.randn(4, 8, dtype=torch.cfloat))
    layer.weight = p
    assert layer.linear.weight is p
    x = torch.randn(2, 8, dtype=torch.cfloat)
    torch.testing.assert_close(layer(x), x @ p.t() + layer.bias)


def test_linear_plain_tensor_assignment_raises_type_error():
    """torch-parity: nn.Linear itself rejects non-Parameter assignment."""
    import pytest

    layer = Linear(8, 4)
    with pytest.raises(TypeError, match="cannot assign"):
        layer.weight = torch.randn(4, 8, dtype=torch.cfloat)


def test_linear_state_dict_keys_unchanged():
    assert set(Linear(8, 4).state_dict()) == {"linear.weight", "linear.bias"}


def test_linear_bias_parameter_assignment_rebinds():
    layer = Linear(8, 4)
    p = torch.nn.Parameter(torch.randn(4, dtype=torch.cfloat))
    layer.bias = p
    assert layer.linear.bias is p
