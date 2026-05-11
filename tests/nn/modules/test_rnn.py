"""Tests for complex GRU/LSTM cells and multi-layer wrappers."""

from __future__ import annotations

import pytest
import torch

from complextorch.nn.modules.rnn import GRU, LSTM, GRUCell, LSTMCell

# ---------- Cells ----------


@pytest.mark.parametrize("batchnorm", [False, True])
def test_grucell(batchnorm):
    cell = GRUCell(input_size=4, hidden_size=6, batchnorm=batchnorm)
    x = torch.randn(8, 4, dtype=torch.cfloat)
    h = cell(x)
    assert h.shape == (8, 6)
    assert h.is_complex()
    # With provided hx
    h2 = cell(x, hx=h)
    assert h2.shape == (8, 6)


@pytest.mark.parametrize("batchnorm", [False, True])
def test_lstmcell(batchnorm):
    cell = LSTMCell(input_size=4, hidden_size=6, batchnorm=batchnorm)
    x = torch.randn(8, 4, dtype=torch.cfloat)
    h, c = cell(x)
    assert h.shape == (8, 6)
    assert c.shape == (8, 6)
    assert h.is_complex()
    assert c.is_complex()
    h2, c2 = cell(x, hx=(h, c))
    assert h2.shape == (8, 6)
    assert c2.shape == (8, 6)


# ---------- Multi-layer ----------


@pytest.mark.parametrize("batch_first", [False, True])
@pytest.mark.parametrize("bidirectional", [False, True])
def test_gru_multilayer(batch_first, bidirectional):
    n_dir = 2 if bidirectional else 1
    gru = GRU(
        input_size=4,
        hidden_size=6,
        num_layers=2,
        batch_first=batch_first,
        dropout=0.1,
        bidirectional=bidirectional,
    )
    if batch_first:
        x = torch.randn(3, 5, 4, dtype=torch.cfloat)  # (B, T, F)
    else:
        x = torch.randn(5, 3, 4, dtype=torch.cfloat)  # (T, B, F)
    out, h = gru(x)
    assert out.is_complex()
    assert h.shape == (2 * n_dir, 3, 6)


def test_gru_with_provided_hx():
    gru = GRU(input_size=4, hidden_size=6, num_layers=1)
    x = torch.randn(5, 3, 4, dtype=torch.cfloat)
    hx = torch.randn(1, 3, 6, dtype=torch.cfloat)
    out, _h = gru(x, hx)
    assert out.is_complex()


@pytest.mark.parametrize("batch_first", [False, True])
@pytest.mark.parametrize("bidirectional", [False, True])
def test_lstm_multilayer(batch_first, bidirectional):
    n_dir = 2 if bidirectional else 1
    lstm = LSTM(
        input_size=4,
        hidden_size=6,
        num_layers=2,
        batch_first=batch_first,
        dropout=0.1,
        bidirectional=bidirectional,
    )
    if batch_first:
        x = torch.randn(3, 5, 4, dtype=torch.cfloat)
    else:
        x = torch.randn(5, 3, 4, dtype=torch.cfloat)
    out, (h, c) = lstm(x)
    assert out.is_complex()
    assert h.shape == (2 * n_dir, 3, 6)
    assert c.shape == (2 * n_dir, 3, 6)


def test_lstm_with_provided_hx():
    lstm = LSTM(input_size=4, hidden_size=6, num_layers=1)
    x = torch.randn(5, 3, 4, dtype=torch.cfloat)
    h0 = torch.randn(1, 3, 6, dtype=torch.cfloat)
    c0 = torch.randn(1, 3, 6, dtype=torch.cfloat)
    out, (_h, _c) = lstm(x, (h0, c0))
    assert out.is_complex()
