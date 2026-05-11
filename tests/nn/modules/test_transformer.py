"""Tests for complex Transformer encoder / decoder / full."""

from __future__ import annotations

import pytest
import torch

from complextorch.nn.modules.transformer import (
    Transformer,
    TransformerDecoder,
    TransformerDecoderLayer,
    TransformerEncoder,
    TransformerEncoderLayer,
)


@pytest.mark.parametrize("activation", ["gelu", "relu"])
def test_encoder_layer_forward(activation):
    layer = TransformerEncoderLayer(
        d_model=8, nhead=2, dim_feedforward=16, activation=activation
    )
    src = torch.randn(2, 5, 8, dtype=torch.cfloat)
    out = layer(src)
    assert out.shape == src.shape


def test_encoder_layer_batch_first_false():
    layer = TransformerEncoderLayer(
        d_model=8, nhead=2, dim_feedforward=16, batch_first=False
    )
    src = torch.randn(5, 2, 8, dtype=torch.cfloat)
    out = layer(src)
    assert out.shape == src.shape


def test_encoder_layer_invalid_d_model_raises():
    with pytest.raises(ValueError, match="must be divisible"):
        TransformerEncoderLayer(d_model=8, nhead=3)


def test_encoder_layer_invalid_activation():
    with pytest.raises(ValueError, match="Unknown activation"):
        TransformerEncoderLayer(d_model=8, nhead=2, activation="bogus")


def test_encoder_stack():
    layer = TransformerEncoderLayer(d_model=8, nhead=2, dim_feedforward=16)
    enc = TransformerEncoder(layer, num_layers=2)
    src = torch.randn(2, 5, 8, dtype=torch.cfloat)
    out = enc(src)
    assert out.shape == src.shape


def test_encoder_stack_with_norm():
    layer = TransformerEncoderLayer(d_model=8, nhead=2, dim_feedforward=16)
    from complextorch.nn.modules.layernorm import LayerNorm

    enc = TransformerEncoder(layer, num_layers=1, norm=LayerNorm(8))
    src = torch.randn(2, 5, 8, dtype=torch.cfloat)
    out = enc(src)
    assert out.shape == src.shape


def test_decoder_layer_forward():
    layer = TransformerDecoderLayer(d_model=8, nhead=2, dim_feedforward=16)
    tgt = torch.randn(2, 5, 8, dtype=torch.cfloat)
    mem = torch.randn(2, 7, 8, dtype=torch.cfloat)
    out = layer(tgt, mem)
    assert out.shape == tgt.shape


def test_decoder_layer_batch_first_false():
    layer = TransformerDecoderLayer(
        d_model=8, nhead=2, dim_feedforward=16, batch_first=False
    )
    tgt = torch.randn(5, 2, 8, dtype=torch.cfloat)
    mem = torch.randn(7, 2, 8, dtype=torch.cfloat)
    out = layer(tgt, mem)
    assert out.shape == tgt.shape


def test_decoder_stack():
    layer = TransformerDecoderLayer(d_model=8, nhead=2, dim_feedforward=16)
    dec = TransformerDecoder(layer, num_layers=2)
    tgt = torch.randn(2, 5, 8, dtype=torch.cfloat)
    mem = torch.randn(2, 7, 8, dtype=torch.cfloat)
    out = dec(tgt, mem)
    assert out.shape == tgt.shape


def test_decoder_stack_with_norm():
    layer = TransformerDecoderLayer(d_model=8, nhead=2, dim_feedforward=16)
    from complextorch.nn.modules.layernorm import LayerNorm

    dec = TransformerDecoder(layer, num_layers=1, norm=LayerNorm(8))
    tgt = torch.randn(2, 5, 8, dtype=torch.cfloat)
    mem = torch.randn(2, 7, 8, dtype=torch.cfloat)
    out = dec(tgt, mem)
    assert out.shape == tgt.shape


def test_full_transformer():
    model = Transformer(
        d_model=8,
        nhead=2,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dim_feedforward=16,
    )
    src = torch.randn(2, 5, 8, dtype=torch.cfloat)
    tgt = torch.randn(2, 4, 8, dtype=torch.cfloat)
    out = model(src, tgt)
    assert out.shape == tgt.shape
