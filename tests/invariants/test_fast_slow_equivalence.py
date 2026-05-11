"""Property tests: Fast/Slow conv and linear produce equivalent output with shared weights."""

from __future__ import annotations

import pytest
import torch
from hypothesis import given, settings, strategies as st

from complextorch.nn.modules.conv import (
    Conv1d,
    Conv2d,
    Conv3d,
    SlowConv1d,
    SlowConv2d,
    SlowConv3d,
)
from complextorch.nn.modules.linear import Linear, SlowLinear


def _align_slow_conv_with_fast(slow, fast):
    with torch.no_grad():
        slow.conv_r.weight.copy_(fast.conv.weight.real)
        slow.conv_i.weight.copy_(fast.conv.weight.imag)
        if fast.conv.bias is not None:
            slow.bias_r.copy_(fast.conv.bias.real)
            slow.bias_i.copy_(fast.conv.bias.imag)


@pytest.mark.parametrize("in_ch,out_ch,k", [(2, 4, 3), (1, 1, 1), (3, 2, 5)])
@given(batch=st.integers(1, 3))
@settings(max_examples=3, deadline=None)
def test_conv1d_fast_slow_equivalence(in_ch, out_ch, k, batch):
    fast = Conv1d(in_ch, out_ch, kernel_size=k, bias=True)
    slow = SlowConv1d(in_ch, out_ch, kernel_size=k, bias=True)
    _align_slow_conv_with_fast(slow, fast)
    x = torch.randn(batch, in_ch, 16, dtype=torch.cfloat)
    torch.testing.assert_close(fast(x), slow(x), rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("in_ch,out_ch,k", [(2, 4, 3), (1, 1, 1)])
@given(batch=st.integers(1, 2))
@settings(max_examples=3, deadline=None)
def test_conv2d_fast_slow_equivalence(in_ch, out_ch, k, batch):
    fast = Conv2d(in_ch, out_ch, kernel_size=k, bias=True)
    slow = SlowConv2d(in_ch, out_ch, kernel_size=k, bias=True)
    _align_slow_conv_with_fast(slow, fast)
    x = torch.randn(batch, in_ch, 8, 8, dtype=torch.cfloat)
    torch.testing.assert_close(fast(x), slow(x), rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("in_ch,out_ch,k", [(2, 2, 3)])
@settings(max_examples=2, deadline=None)
@given(batch=st.integers(1, 2))
def test_conv3d_fast_slow_equivalence(in_ch, out_ch, k, batch):
    fast = Conv3d(in_ch, out_ch, kernel_size=k, bias=True)
    slow = SlowConv3d(in_ch, out_ch, kernel_size=k, bias=True)
    _align_slow_conv_with_fast(slow, fast)
    x = torch.randn(batch, in_ch, 4, 4, 4, dtype=torch.cfloat)
    torch.testing.assert_close(fast(x), slow(x), rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("in_f,out_f", [(6, 8), (1, 1), (12, 5)])
@given(batch=st.integers(1, 4))
@settings(max_examples=3, deadline=None)
def test_linear_fast_slow_equivalence(in_f, out_f, batch):
    fast = Linear(in_f, out_f, bias=True)
    slow = SlowLinear(in_f, out_f, bias=True)
    with torch.no_grad():
        slow.linear_r.weight.copy_(fast.linear.weight.real)
        slow.linear_i.weight.copy_(fast.linear.weight.imag)
        slow.bias_r.copy_(fast.linear.bias.real)
        slow.bias_i.copy_(fast.linear.bias.imag)
    x = torch.randn(batch, in_f, dtype=torch.cfloat)
    torch.testing.assert_close(fast(x), slow(x), rtol=1e-4, atol=1e-4)
