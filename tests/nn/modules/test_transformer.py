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
        d_model=8,
        nhead=2,
        dim_feedforward=16,
        activation=activation,
        batch_first=True,
    )
    src = torch.randn(2, 5, 8, dtype=torch.cfloat)
    out = layer(src)
    assert out.shape == src.shape


def test_encoder_layer_batch_first_false_is_default():
    """torch.nn.Transformer parity: sequence-first is the default layout."""
    layer = TransformerEncoderLayer(d_model=8, nhead=2, dim_feedforward=16)
    assert layer.batch_first is False
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
    layer = TransformerEncoderLayer(
        d_model=8, nhead=2, dim_feedforward=16, batch_first=True
    )
    enc = TransformerEncoder(layer, num_layers=2)
    src = torch.randn(2, 5, 8, dtype=torch.cfloat)
    out = enc(src)
    assert out.shape == src.shape


def test_encoder_stack_with_norm():
    layer = TransformerEncoderLayer(
        d_model=8, nhead=2, dim_feedforward=16, batch_first=True
    )
    from complextorch.nn.modules.layernorm import LayerNorm

    enc = TransformerEncoder(layer, num_layers=1, norm=LayerNorm(8))
    src = torch.randn(2, 5, 8, dtype=torch.cfloat)
    out = enc(src)
    assert out.shape == src.shape


def test_decoder_layer_forward():
    layer = TransformerDecoderLayer(
        d_model=8, nhead=2, dim_feedforward=16, batch_first=True
    )
    tgt = torch.randn(2, 5, 8, dtype=torch.cfloat)
    mem = torch.randn(2, 7, 8, dtype=torch.cfloat)
    out = layer(tgt, mem)
    assert out.shape == tgt.shape


def test_decoder_layer_batch_first_false():
    layer = TransformerDecoderLayer(d_model=8, nhead=2, dim_feedforward=16)
    assert layer.batch_first is False
    tgt = torch.randn(5, 2, 8, dtype=torch.cfloat)
    mem = torch.randn(7, 2, 8, dtype=torch.cfloat)
    out = layer(tgt, mem)
    assert out.shape == tgt.shape


def test_decoder_stack():
    layer = TransformerDecoderLayer(
        d_model=8, nhead=2, dim_feedforward=16, batch_first=True
    )
    dec = TransformerDecoder(layer, num_layers=2)
    tgt = torch.randn(2, 5, 8, dtype=torch.cfloat)
    mem = torch.randn(2, 7, 8, dtype=torch.cfloat)
    out = dec(tgt, mem)
    assert out.shape == tgt.shape


def test_decoder_stack_with_norm():
    layer = TransformerDecoderLayer(
        d_model=8, nhead=2, dim_feedforward=16, batch_first=True
    )
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
        batch_first=True,
    )
    src = torch.randn(2, 5, 8, dtype=torch.cfloat)
    tgt = torch.randn(2, 4, 8, dtype=torch.cfloat)
    out = model(src, tgt)
    assert out.shape == tgt.shape


# ---------- Masks & causality ----------


def test_generate_square_subsequent_mask_contents():
    m = Transformer.generate_square_subsequent_mask(4)
    assert m.shape == (4, 4)
    assert m.dtype == torch.get_default_dtype()
    for i in range(4):
        for j in range(4):
            if j > i:
                assert m[i, j] == float("-inf")
            else:
                assert m[i, j] == 0.0


def test_generate_square_subsequent_mask_dtype_device_kwargs():
    m = Transformer.generate_square_subsequent_mask(
        3, device=torch.device("cpu"), dtype=torch.float64
    )
    assert m.dtype == torch.float64
    assert m.device.type == "cpu"


def test_decoder_layer_invalid_d_model_raises():
    """Parity with the encoder layer's divisibility guard."""
    with pytest.raises(ValueError, match="must be divisible"):
        TransformerDecoderLayer(d_model=8, nhead=3)


def _causal_independence(outputs_fn, seq_len):
    """Perturbing the last position must leave earlier outputs unchanged."""
    torch.manual_seed(0)
    x = torch.randn(1, seq_len, 8, dtype=torch.cfloat)
    out = outputs_fn(x)
    x2 = x.clone()
    x2[:, -1] += 10.0 + 5.0j  # perturb the last position
    out2 = outputs_fn(x2)
    torch.testing.assert_close(out[:, :-1], out2[:, :-1])


def test_encoder_layer_causal_src_mask():
    layer = TransformerEncoderLayer(
        d_model=8, nhead=2, dim_feedforward=16, dropout=0.0, batch_first=True
    )
    layer.eval()
    mask = Transformer.generate_square_subsequent_mask(6)
    _causal_independence(lambda x: layer(x, src_mask=mask), 6)


def test_encoder_stack_key_padding_mask_equals_truncated_input():
    layer = TransformerEncoderLayer(
        d_model=8, nhead=2, dim_feedforward=16, dropout=0.0, batch_first=True
    )
    enc = TransformerEncoder(layer, num_layers=2)
    enc.eval()
    torch.manual_seed(0)
    x = torch.randn(1, 6, 8, dtype=torch.cfloat)
    kpm = torch.zeros(1, 6, dtype=torch.bool)
    kpm[:, 4:] = True
    out = enc(x, src_key_padding_mask=kpm)
    expected = enc(x[:, :4])
    torch.testing.assert_close(out[:, :4], expected)


def test_decoder_layer_causal_tgt_mask():
    layer = TransformerDecoderLayer(
        d_model=8, nhead=2, dim_feedforward=16, dropout=0.0, batch_first=True
    )
    layer.eval()
    torch.manual_seed(1)
    mem = torch.randn(1, 5, 8, dtype=torch.cfloat)
    mask = Transformer.generate_square_subsequent_mask(6)
    _causal_independence(lambda x: layer(x, mem, tgt_mask=mask), 6)


def test_decoder_layer_memory_and_padding_masks_smoke():
    layer = TransformerDecoderLayer(
        d_model=8, nhead=2, dim_feedforward=16, dropout=0.0, batch_first=True
    )
    layer.eval()
    tgt = torch.randn(2, 4, 8, dtype=torch.cfloat)
    mem = torch.randn(2, 5, 8, dtype=torch.cfloat)
    out = layer(
        tgt,
        mem,
        tgt_mask=Transformer.generate_square_subsequent_mask(4),
        memory_mask=torch.zeros(4, 5, dtype=torch.bool),
        tgt_key_padding_mask=torch.zeros(2, 4, dtype=torch.bool),
        memory_key_padding_mask=torch.zeros(2, 5, dtype=torch.bool),
    )
    assert out.shape == tgt.shape
    assert torch.isfinite(out.real).all()


def test_full_transformer_all_masks_and_causality():
    model = Transformer(
        d_model=8,
        nhead=2,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dim_feedforward=16,
        dropout=0.0,
        batch_first=True,
    )
    model.eval()
    torch.manual_seed(0)
    src = torch.randn(1, 5, 8, dtype=torch.cfloat)
    tgt_mask = Transformer.generate_square_subsequent_mask(6)

    def run(tgt):
        return model(
            src,
            tgt,
            src_mask=torch.zeros(5, 5, dtype=torch.bool),
            tgt_mask=tgt_mask,
            memory_mask=torch.zeros(6, 5, dtype=torch.bool),
            src_key_padding_mask=torch.zeros(1, 5, dtype=torch.bool),
            tgt_key_padding_mask=torch.zeros(1, 6, dtype=torch.bool),
            memory_key_padding_mask=torch.zeros(1, 5, dtype=torch.bool),
        )

    _causal_independence(run, 6)


def test_transformer_default_is_seq_first():
    """torch.nn.Transformer parity: batch_first defaults to False."""
    model = Transformer(
        d_model=8,
        nhead=2,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dim_feedforward=16,
    )
    assert model.batch_first is False
    src = torch.randn(5, 2, 8, dtype=torch.cfloat)  # (T, B, F)
    tgt = torch.randn(4, 2, 8, dtype=torch.cfloat)
    out = model(src, tgt)
    assert out.shape == tgt.shape
