"""Tests for complex Dropout / Dropout{1,2,3}d."""

from __future__ import annotations

import pytest
import torch

from complextorch.nn.modules.dropout import Dropout, Dropout1d, Dropout2d, Dropout3d


def test_dropout_eval_passthrough():
    drop = Dropout(p=0.9)
    drop.eval()
    x = torch.randn(4, 8, dtype=torch.cfloat)
    out = drop(x)
    torch.testing.assert_close(out, x)


def test_dropout_train_changes_inputs():
    drop = Dropout(p=0.5)
    drop.train()
    x = torch.ones(1000, dtype=torch.cfloat)
    out = drop(x)
    # Many entries should now be zero
    assert (out == 0).any()


def test_dropout1d_eval_is_identity():
    drop = Dropout1d(p=0.5)
    drop.eval()
    x = torch.randn(2, 4, 8, dtype=torch.cfloat)
    out = drop(x)
    torch.testing.assert_close(out, x)


@pytest.mark.parametrize(
    ("cls", "shape"),
    [
        (Dropout1d, (2, 4, 8)),
        (Dropout2d, (2, 4, 4, 4)),
        (Dropout3d, (1, 4, 3, 3, 3)),
    ],
)
def test_channel_dropout_complex_train(cls, shape):
    drop = cls(p=0.5)
    drop.train()
    x = torch.randn(*shape, dtype=torch.cfloat)
    out = drop(x)
    assert out.shape == x.shape
    assert out.is_complex()


def test_channel_dropout_p_zero_is_identity():
    drop = Dropout1d(p=0.0)
    drop.train()
    x = torch.randn(2, 4, 8, dtype=torch.cfloat)
    out = drop(x)
    torch.testing.assert_close(out, x)


def test_channel_dropout_invalid_p():
    with pytest.raises(ValueError, match="must be in"):
        Dropout1d(p=1.0)


def test_channel_dropout_real_input_path():
    drop = Dropout1d(p=0.5)
    drop.train()
    x = torch.randn(2, 4, 8)
    out = drop(x)
    assert out.shape == x.shape


def test_channel_dropout_extra_repr():
    s = Dropout2d(p=0.3).extra_repr()
    assert "p=0.3" in s
