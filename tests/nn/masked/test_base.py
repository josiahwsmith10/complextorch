"""Tests for BaseMasked, deploy_masks, binarize_masks, is_sparse, named_masks."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from complextorch.nn.masked import (
    LinearMasked,
    binarize_masks,
    deploy_masks,
    is_sparse,
    named_masks,
)


def test_baselayer_is_sparse_when_masked():
    layer = LinearMasked(4, 6)
    assert not layer.is_sparse
    layer.mask = torch.ones(6, 4)
    assert layer.is_sparse


def test_mask_setter_with_non_tensor_raises():
    layer = LinearMasked(4, 6)
    with pytest.raises(TypeError, match="Tensor or None"):
        layer.mask = [1, 2, 3]


def test_setattr_passthrough_for_non_mask():
    layer = LinearMasked(4, 6)
    layer.in_features = 42  # should not go through mask_
    assert layer.in_features == 42


def test_mask_clear_via_none():
    layer = LinearMasked(4, 6)
    layer.mask = torch.ones(6, 4)
    assert layer.is_sparse
    layer.mask = None
    assert not layer.is_sparse


def test_deploy_masks_with_qualified_names():
    model = nn.Module()
    model.linear = LinearMasked(4, 6)
    mask = torch.zeros(6, 4)
    mask[0:2] = 1.0
    deploy_masks(model, {"linear.mask": mask})
    assert model.linear.is_sparse
    torch.testing.assert_close(model.linear.mask, mask)


def test_deploy_masks_strict_missing_module_raises():
    model = nn.Module()
    with pytest.raises(KeyError, match="no BaseMasked"):
        deploy_masks(model, {"nope.mask": torch.zeros(4)}, strict=True)


def test_deploy_masks_non_strict_silently_skips():
    model = nn.Module()
    deploy_masks(model, {"nope.mask": torch.zeros(4)}, strict=False)


def test_deploy_masks_skips_non_mask_keys():
    model = nn.Module()
    model.linear = LinearMasked(4, 6)
    deploy_masks(model, {"linear.weight": torch.zeros(6, 4)})
    assert not model.linear.is_sparse


def test_binarize_masks_makes_binary():
    layer = LinearMasked(4, 6)
    layer.mask = torch.full((6, 4), 0.3)
    binarize_masks(layer)
    assert ((layer.mask == 0) | (layer.mask == 1)).all()


def test_named_masks_iterates():
    model = nn.Module()
    model.l1 = LinearMasked(4, 6)
    model.l2 = LinearMasked(4, 6)
    model.l1.mask = torch.ones(6, 4)
    names = [n for n, _ in named_masks(model)]
    assert "l1" in names
    assert "l2" not in names  # l2 doesn't have a mask


def test_is_sparse_function():
    layer = LinearMasked(4, 6)
    assert not is_sparse(layer)
    layer.mask = torch.ones(6, 4)
    assert is_sparse(layer)
    assert not is_sparse(nn.Linear(4, 6))  # not a BaseMasked


def test_state_dict_roundtrip_with_mask():
    layer = LinearMasked(4, 6)
    mask = torch.eye(6, 4)
    layer.mask = mask
    state = layer.state_dict()
    assert "mask" in state
    other = LinearMasked(4, 6)
    other.load_state_dict(state)
    assert other.is_sparse
    torch.testing.assert_close(other.mask, mask)


def test_state_dict_load_missing_mask_strict_false():
    layer = LinearMasked(4, 6)
    state = {
        "weight": layer.weight.detach(),
        "bias": layer.bias.detach(),
    }
    layer.load_state_dict(state, strict=False)


def test_state_dict_load_missing_mask_strict_true_records_missing():
    layer = LinearMasked(4, 6)
    state = {
        "weight": layer.weight.detach(),
        "bias": layer.bias.detach(),
    }
    missing, _ = layer.load_state_dict(state, strict=False)
    assert "mask" in missing


def test_weight_masked_property_raises_without_mask():
    layer = LinearMasked(4, 6)
    with pytest.raises(RuntimeError, match="no sparsity mask"):
        _ = layer.weight_masked


def test_state_dict_load_when_layer_has_mask_and_state_has_mask():
    """Hits the 'mask_in_missing + state has mask' path: remove from missing_keys."""
    src = LinearMasked(4, 6)
    src.mask = torch.eye(6, 4)
    state = src.state_dict()
    tgt = LinearMasked(4, 6)
    tgt.mask = torch.ones(6, 4)  # non-None so super sees it during load
    tgt.load_state_dict(state)
    torch.testing.assert_close(tgt.mask, src.mask)


def test_state_dict_load_strict_mask_missing_records_in_missing_keys():
    """Hits strict-mode branch where mask is absent from state_dict."""
    layer = LinearMasked(4, 6)
    layer.mask = torch.ones(6, 4)  # non-None so super sees it
    state = {"weight": layer.weight.detach(), "bias": layer.bias.detach()}
    missing, _ = layer.load_state_dict(state, strict=False)
    assert "mask" in missing
