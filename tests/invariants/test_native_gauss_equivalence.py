"""Property tests: native cfloat and Gauss-trick conv/linear agree given shared weights."""

from __future__ import annotations

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from complextorch.nn.gauss.conv import (
    Conv1d as GaussConv1d,
)
from complextorch.nn.gauss.conv import (
    Conv2d as GaussConv2d,
)
from complextorch.nn.gauss.conv import (
    Conv3d as GaussConv3d,
)
from complextorch.nn.gauss.linear import Linear as GaussLinear
from complextorch.nn.modules.conv import Conv1d, Conv2d, Conv3d
from complextorch.nn.modules.linear import Linear


def _align_gauss_conv_with_native(gauss, native):
    with torch.no_grad():
        gauss.conv_r.weight.copy_(native.conv.weight.real)
        gauss.conv_i.weight.copy_(native.conv.weight.imag)
        if native.conv.bias is not None:
            gauss.bias_r.copy_(native.conv.bias.real)
            gauss.bias_i.copy_(native.conv.bias.imag)


@pytest.mark.parametrize(("in_ch", "out_ch", "k"), [(2, 4, 3), (1, 1, 1), (3, 2, 5)])
@given(batch=st.integers(1, 3))
@settings(max_examples=3, deadline=None)
def test_conv1d_native_gauss_equivalence(in_ch, out_ch, k, batch):
    native = Conv1d(in_ch, out_ch, kernel_size=k, bias=True)
    gauss = GaussConv1d(in_ch, out_ch, kernel_size=k, bias=True)
    _align_gauss_conv_with_native(gauss, native)
    x = torch.randn(batch, in_ch, 16, dtype=torch.cfloat)
    torch.testing.assert_close(native(x), gauss(x), rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize(("in_ch", "out_ch", "k"), [(2, 4, 3), (1, 1, 1)])
@given(batch=st.integers(1, 2))
@settings(max_examples=3, deadline=None)
def test_conv2d_native_gauss_equivalence(in_ch, out_ch, k, batch):
    native = Conv2d(in_ch, out_ch, kernel_size=k, bias=True)
    gauss = GaussConv2d(in_ch, out_ch, kernel_size=k, bias=True)
    _align_gauss_conv_with_native(gauss, native)
    x = torch.randn(batch, in_ch, 8, 8, dtype=torch.cfloat)
    torch.testing.assert_close(native(x), gauss(x), rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize(("in_ch", "out_ch", "k"), [(2, 2, 3)])
@settings(max_examples=2, deadline=None)
@given(batch=st.integers(1, 2))
def test_conv3d_native_gauss_equivalence(in_ch, out_ch, k, batch):
    native = Conv3d(in_ch, out_ch, kernel_size=k, bias=True)
    gauss = GaussConv3d(in_ch, out_ch, kernel_size=k, bias=True)
    _align_gauss_conv_with_native(gauss, native)
    x = torch.randn(batch, in_ch, 4, 4, 4, dtype=torch.cfloat)
    torch.testing.assert_close(native(x), gauss(x), rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize(("in_f", "out_f"), [(6, 8), (1, 1), (12, 5)])
@given(batch=st.integers(1, 4))
@settings(max_examples=3, deadline=None)
def test_linear_native_gauss_equivalence(in_f, out_f, batch):
    native = Linear(in_f, out_f, bias=True)
    gauss = GaussLinear(in_f, out_f, bias=True)
    with torch.no_grad():
        gauss.linear_r.weight.copy_(native.linear.weight.real)
        gauss.linear_i.weight.copy_(native.linear.weight.imag)
        gauss.bias_r.copy_(native.linear.bias.real)
        gauss.bias_i.copy_(native.linear.bias.imag)
    x = torch.randn(batch, in_f, dtype=torch.cfloat)
    torch.testing.assert_close(native(x), gauss(x), rtol=1e-4, atol=1e-4)
