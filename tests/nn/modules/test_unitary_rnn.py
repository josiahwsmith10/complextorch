"""Tests for unitary complex RNNs."""

from __future__ import annotations

import torch

from complextorch.nn.modules.unitary_rnn import UnitaryRNN, UnitaryRNNCell


def test_unitary_matrix_is_unitary():
    cell = UnitaryRNNCell(4, 6)
    w = cell.unitary_matrix()
    identity = torch.eye(6, dtype=torch.cfloat)
    torch.testing.assert_close(
        w.conj().transpose(-2, -1) @ w, identity, atol=1e-5, rtol=1e-5
    )


def test_unitary_recurrence_preserves_norm():
    cell = UnitaryRNNCell(4, 6)
    w = cell.unitary_matrix()
    h = torch.randn(3, 6, dtype=torch.cfloat)
    wh = h @ w.transpose(-2, -1)
    torch.testing.assert_close(
        wh.abs().pow(2).sum(-1), h.abs().pow(2).sum(-1), atol=1e-4, rtol=1e-4
    )


def test_unitary_cell_forward_default_state():
    cell = UnitaryRNNCell(4, 6)
    x = torch.randn(3, 4, dtype=torch.cfloat)
    out = cell(x)
    assert out.shape == (3, 6)
    assert out.is_complex()


def test_unitary_cell_grad_flows_into_generator():
    cell = UnitaryRNNCell(4, 6)
    x = torch.randn(3, 4, dtype=torch.cfloat)
    h0 = torch.randn(3, 6, dtype=torch.cfloat)  # nonzero so W is exercised
    cell(x, h0).abs().sum().backward()
    assert cell.M.grad is not None
    assert torch.isfinite(cell.M.grad).all()


def test_unitary_rnn_sequence_shapes():
    rnn = UnitaryRNN(4, 6, num_layers=2, dropout=0.1)
    x = torch.randn(5, 3, 4, dtype=torch.cfloat)  # (T, B, F)
    out, h_n = rnn(x)
    assert out.shape == (5, 3, 6)
    assert h_n.shape == (2, 3, 6)


def test_unitary_rnn_batch_first_and_given_state():
    rnn = UnitaryRNN(4, 6, batch_first=True)
    x = torch.randn(3, 5, 4, dtype=torch.cfloat)  # (B, T, F)
    h0 = torch.zeros(1, 3, 6, dtype=torch.cfloat)
    out, h_n = rnn(x, h0)
    assert out.shape == (3, 5, 6)
    assert h_n.shape == (1, 3, 6)


def test_unitary_rnn_grad_flows():
    rnn = UnitaryRNN(4, 6, num_layers=1)
    x = torch.randn(5, 2, 4, dtype=torch.cfloat)
    out, _ = rnn(x)
    out.abs().sum().backward()
    assert rnn.cells[0].M.grad is not None
