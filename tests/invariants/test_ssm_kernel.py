"""Property test: the SSM FFT convolution equals the recurrent rollout.

The diagonal state-space layers compute their output two ways -- an FFT long
convolution with the materialised kernel (:meth:`forward`) and a step-by-step
recurrence (:meth:`recurrence`). They must agree up to floating-point error.
"""

from __future__ import annotations

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from complextorch.nn.modules.ssm import DSS, S4D


@pytest.mark.parametrize("cls", [S4D, DSS])
@given(seed=st.integers(0, 10_000), length=st.integers(4, 24))
@settings(max_examples=5, deadline=None)
def test_ssm_kernel_matches_recurrence(cls, seed, length):
    torch.manual_seed(seed)
    model = cls(channels=3, state_size=8)
    u = torch.randn(2, length, 3, dtype=torch.cfloat)
    y_fft = model(u)
    y_rec = model.recurrence(u)
    torch.testing.assert_close(y_fft, y_rec, atol=1e-4, rtol=1e-4)
