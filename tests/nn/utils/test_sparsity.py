"""Tests for SparsityStats / named_sparsity / sparsity."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from complextorch.nn.masked import LinearMasked
from complextorch.nn.utils import SparsityStats, named_sparsity, sparsity


def test_sparsity_stats_default_raises():
    class M(SparsityStats):
        pass

    m = M()
    with pytest.raises(NotImplementedError):
        m.sparsity()


def test_named_sparsity_no_subscriber_yields_nothing():
    model = nn.Linear(4, 6)
    pairs = list(named_sparsity(model))
    assert pairs == []


def test_named_sparsity_on_masked_layer():
    model = nn.Module()
    model.l1 = LinearMasked(4, 6)
    model.l1.mask = torch.zeros(6, 4)
    pairs = list(named_sparsity(model))
    assert len(pairs) == 1
    name, (n_zeros, n_total) = pairs[0]
    assert "weight" in name
    assert n_zeros == 24  # 6*4 weights, all zero
    assert n_total == 24


def test_sparsity_global():
    model = nn.Module()
    model.l1 = LinearMasked(4, 6)
    model.l1.mask = torch.zeros(6, 4)
    s = sparsity(model)
    assert s == 1.0


def test_sparsity_no_params_returns_zero():
    """Empty module -> 0/0 -> 0.0 via the `if total` guard."""
    model = nn.Module()
    s = sparsity(model)
    assert s == 0.0


def test_named_sparsity_dense_masked_layer_reports_zero():
    """LinearMasked without a mask set -> n_dropped == 0."""
    model = nn.Module()
    model.l1 = LinearMasked(4, 6)
    pairs = list(named_sparsity(model))
    assert len(pairs) == 1
    _, (n_zeros, n_total) = pairs[0]
    assert n_zeros == 0
    assert n_total == 24


def test_named_sparsity_skips_duplicate_pids_and_unknown_pids():
    """Hit the 'pid in seen or pid not in pid_to_name: continue' branch."""

    class DupAndForeign(SparsityStats):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(3))

        def sparsity(self, **kwargs):
            # Real pid twice + a foreign pid.
            return [
                (id(self.weight), 1),
                (id(self.weight), 1),  # duplicate -> 'pid in seen' branch
                (99999999, 7),  # foreign -> 'pid not in pid_to_name' branch
            ]

    model = nn.Module()
    model.s = DupAndForeign()
    pairs = list(named_sparsity(model))
    # Only the unique, known pid yields a result.
    assert len(pairs) == 1
