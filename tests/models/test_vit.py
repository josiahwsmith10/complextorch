"""Tests for the complex Vision Transformer (ViT) and preset factories."""

from __future__ import annotations

import pytest
import torch

from complextorch.models import ViT, ViTLayer, vit_b, vit_h, vit_l, vit_s, vit_t

# ---------- ViTLayer ----------


def test_vit_layer_forward():
    layer = ViTLayer(dim=8, nhead=2, mlp_dim=16)
    x = torch.randn(2, 5, 8, dtype=torch.cfloat)
    out = layer(x)
    assert out.shape == x.shape
    assert out.is_complex()


def test_vit_layer_invalid_dim():
    with pytest.raises(ValueError, match="must be divisible"):
        ViTLayer(dim=8, nhead=3, mlp_dim=16)


# ---------- ViT (full) ----------


def test_vit_forward_tiny():
    vit = ViT(
        image_size=16,
        patch_size=4,
        in_channels=1,
        num_classes=5,
        dim=16,
        depth=1,
        heads=2,
        mlp_dim=32,
    )
    x = torch.randn(1, 1, 16, 16, dtype=torch.cfloat)
    out = vit(x)
    assert out.shape == (1, 5)
    assert out.is_complex()


def test_vit_no_head():
    vit = ViT(
        image_size=16, patch_size=4, num_classes=0, dim=16, depth=1, heads=2, mlp_dim=32
    )
    x = torch.randn(1, 1, 16, 16, dtype=torch.cfloat)
    out = vit(x)
    # nn.Identity head -> dim still complex
    assert out.shape == (1, 16)


def test_vit_invalid_image_patch():
    with pytest.raises(ValueError, match="must be divisible"):
        ViT(
            image_size=17,
            patch_size=4,
            num_classes=5,
            dim=8,
            depth=1,
            heads=2,
            mlp_dim=16,
        )


@pytest.mark.parametrize("pos_encoding", ["learned", "sinusoidal", "rotary"])
def test_vit_pos_encoding_modes(pos_encoding):
    vit = ViT(
        image_size=16,
        patch_size=4,
        num_classes=5,
        dim=16,
        depth=1,
        heads=2,
        mlp_dim=32,
        pos_encoding=pos_encoding,
    )
    x = torch.randn(1, 1, 16, 16, dtype=torch.cfloat)
    out = vit(x)
    assert out.shape == (1, 5)


def test_vit_invalid_pos_encoding():
    with pytest.raises(ValueError, match="pos_encoding must be"):
        ViT(
            image_size=16,
            patch_size=4,
            num_classes=5,
            dim=16,
            depth=1,
            heads=2,
            mlp_dim=32,
            pos_encoding="bogus",
        )


def test_vit_invalid_dim_heads():
    with pytest.raises(ValueError, match="must be divisible by heads"):
        ViT(
            image_size=16,
            patch_size=4,
            num_classes=5,
            dim=18,
            depth=1,
            heads=4,
            mlp_dim=32,
        )


# ---------- Presets (smoke: just instantiate; don't run forward on full sizes) ----------


@pytest.mark.parametrize("factory", [vit_t, vit_s, vit_b, vit_l, vit_h])
def test_vit_factories_instantiate(factory):
    """Use small image_size to keep memory reasonable."""
    model = factory(
        image_size=14 if factory is vit_h else 16,
        patch_size=14 if factory is vit_h else 16,
        num_classes=0,
    )
    # Just confirm it's a ViT.
    assert isinstance(model, ViT)
