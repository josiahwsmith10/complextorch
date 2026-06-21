"""Tests for the CVKAN model."""

from __future__ import annotations

import pytest
import torch

from complextorch.models.kan import CVKAN


def test_cvkan_forward_shape():
    model = CVKAN([4, 8, 3], num_grid=5)
    x = torch.randn(6, 4, dtype=torch.cfloat)
    out = model(x)
    assert out.shape == (6, 3)
    assert out.is_complex()


def test_cvkan_grad_flows():
    model = CVKAN([4, 6, 2], num_grid=4)
    x = torch.randn(6, 4, dtype=torch.cfloat)
    model(x).abs().sum().backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert all(g is not None for g in grads)


def test_cvkan_requires_two_sizes():
    with pytest.raises(ValueError, match="at least two entries"):
        CVKAN([4])
