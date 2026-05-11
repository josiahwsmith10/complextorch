"""Tests for BaseARD + named_penalties / named_relevance / compute_ard_masks."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from complextorch.nn.relevance import (
    BaseARD,
    LinearVD,
    compute_ard_masks,
    named_penalties,
    named_relevance,
    penalties,
)


def test_base_ard_penalty_default_raises():
    base = BaseARD()
    with pytest.raises(NotImplementedError):
        _ = base.penalty


def test_base_ard_relevance_default_raises():
    base = BaseARD()
    with pytest.raises(NotImplementedError):
        base.relevance(threshold=0.0)


def test_named_penalties_sum():
    model = nn.Module()
    model.l1 = LinearVD(4, 6)
    pairs = list(named_penalties(model, reduction="sum"))
    assert pairs[0][0] == "l1"
    assert pairs[0][1].dim() == 0


def test_named_penalties_mean():
    model = nn.Module()
    model.l1 = LinearVD(4, 6)
    pairs = list(named_penalties(model, reduction="mean"))
    assert pairs[0][1].dim() == 0


def test_named_penalties_no_reduction():
    model = nn.Module()
    model.l1 = LinearVD(4, 6)
    pairs = list(named_penalties(model, reduction=None))
    assert pairs[0][1].shape == (6, 4)


def test_named_penalties_invalid_reduction():
    model = nn.Module()
    model.l1 = LinearVD(4, 6)
    with pytest.raises(ValueError, match="reduction must be"):
        list(named_penalties(model, reduction="bogus"))


def test_penalties_generator():
    model = nn.Module()
    model.l1 = LinearVD(4, 6)
    ps = list(penalties(model))
    assert len(ps) == 1


def test_named_relevance():
    model = nn.Module()
    model.l1 = LinearVD(4, 6)
    pairs = list(named_relevance(model, threshold=0.0))
    assert pairs[0][0] == "l1"
    assert pairs[0][1].shape == (6, 4)


def test_compute_ard_masks():
    model = nn.Module()
    model.l1 = LinearVD(4, 6)
    masks = compute_ard_masks(model, threshold=0.0)
    assert "l1.mask" in masks
    assert masks["l1.mask"].shape == (6, 4)


def test_compute_ard_masks_non_module_returns_empty():
    """Guard branch: non-Module input returns empty dict."""
    assert compute_ard_masks("not a module", threshold=0.0) == {}


def test_compute_ard_masks_root_layer():
    """Mask key for a root-level ARD layer is just 'mask'."""
    layer = LinearVD(4, 6)
    masks = compute_ard_masks(layer, threshold=0.0)
    assert "mask" in masks
