r"""
Complex-Valued Transformer
==========================

Encoder / decoder layers and full encoder-decoder transformer for complex
inputs. Composed from existing complex primitives.

Note on building blocks: this library's :class:`MultiheadAttention` is
implemented as a *complete* attention sub-block — it applies QKV projections,
the attention mechanism, a residual connection, and a final
:class:`LayerNorm` internally. The encoder/decoder layers below therefore use
``MultiheadAttention`` directly as the "attention" sub-layer, and only add
the feed-forward sub-block on top with its own residual + LayerNorm.
"""

import copy
from typing import Optional

import torch
import torch.nn as nn

from complextorch.nn.modules.activation.complex_relu import CReLU
from complextorch.nn.modules.activation.split_type_A import CGELU
from complextorch.nn.modules.attention import MultiheadAttention
from complextorch.nn.modules.dropout import Dropout
from complextorch.nn.modules.layernorm import LayerNorm
from complextorch.nn.modules.linear import Linear

__all__ = [
    "TransformerEncoderLayer",
    "TransformerEncoder",
    "TransformerDecoderLayer",
    "TransformerDecoder",
    "Transformer",
]


def _get_activation(name: str) -> nn.Module:
    if name == "gelu":
        return CGELU()
    if name == "relu":
        return CReLU()
    raise ValueError(f"Unknown activation {name!r}; expected 'gelu' or 'relu'")


def _clones(module: nn.Module, n: int) -> nn.ModuleList:
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


class _FFNBlock(nn.Module):
    """Feed-forward sub-block: Linear -> activation -> Linear, with residual + LayerNorm."""

    def __init__(
        self,
        d_model: int,
        dim_feedforward: int,
        activation: str,
        dropout: float,
        layer_norm_eps: float,
    ) -> None:
        super().__init__()
        self.linear1 = Linear(d_model, dim_feedforward)
        self.linear2 = Linear(dim_feedforward, d_model)
        self.activation = _get_activation(activation)
        self.norm = LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout = Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.linear2(self.activation(self.linear1(x)))
        return self.norm(self.dropout(out) + x)


class TransformerEncoderLayer(nn.Module):
    r"""
    Single complex-valued transformer encoder layer.

    Self-attention + feed-forward, each with internal residual + LayerNorm.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "gelu",
        layer_norm_eps: float = 1e-5,
        batch_first: bool = True,
        softmax_on: str = "complex",
    ) -> None:
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by nhead ({nhead})"
            )
        self.batch_first = batch_first
        d_head = d_model // nhead

        self.self_attn = MultiheadAttention(
            nhead, d_model, d_head, d_head, dropout=dropout, softmax_on=softmax_on
        )
        self.ffn = _FFNBlock(
            d_model, dim_feedforward, activation, dropout, layer_norm_eps
        )
        self.d_model = d_model
        self.nhead = nhead

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        x = src if self.batch_first else src.transpose(0, 1)
        x = self.self_attn(x, x, x)
        x = self.ffn(x)
        return x if self.batch_first else x.transpose(0, 1)


class TransformerEncoder(nn.Module):
    r"""Stack of :class:`TransformerEncoderLayer` blocks."""

    def __init__(
        self,
        encoder_layer: TransformerEncoderLayer,
        num_layers: int,
        norm: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.layers = _clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        x = src
        for layer in self.layers:
            x = layer(x)
        if self.norm is not None:
            x = self.norm(x)
        return x


class TransformerDecoderLayer(nn.Module):
    r"""
    Single complex-valued transformer decoder layer.

    Self-attention + cross-attention + feed-forward, each with internal
    residual + LayerNorm.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "gelu",
        layer_norm_eps: float = 1e-5,
        batch_first: bool = True,
        softmax_on: str = "complex",
    ) -> None:
        super().__init__()
        self.batch_first = batch_first
        d_head = d_model // nhead

        self.self_attn = MultiheadAttention(
            nhead, d_model, d_head, d_head, dropout=dropout, softmax_on=softmax_on
        )
        self.cross_attn = MultiheadAttention(
            nhead, d_model, d_head, d_head, dropout=dropout, softmax_on=softmax_on
        )
        self.ffn = _FFNBlock(
            d_model, dim_feedforward, activation, dropout, layer_norm_eps
        )

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        x = tgt if self.batch_first else tgt.transpose(0, 1)
        m = memory if self.batch_first else memory.transpose(0, 1)
        x = self.self_attn(x, x, x)
        x = self.cross_attn(x, m, m)
        x = self.ffn(x)
        return x if self.batch_first else x.transpose(0, 1)


class TransformerDecoder(nn.Module):
    r"""Stack of :class:`TransformerDecoderLayer` blocks."""

    def __init__(
        self,
        decoder_layer: TransformerDecoderLayer,
        num_layers: int,
        norm: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.layers = _clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        x = tgt
        for layer in self.layers:
            x = layer(x, memory)
        if self.norm is not None:
            x = self.norm(x)
        return x


class Transformer(nn.Module):
    r"""
    Complex-Valued Transformer (encoder-decoder).

    Mirrors :class:`torch.nn.Transformer`. ``forward(src, tgt)`` runs the
    encoder on ``src`` and the decoder on ``tgt`` with the encoder output as
    cross-attention memory.
    """

    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "gelu",
        layer_norm_eps: float = 1e-5,
        batch_first: bool = True,
        softmax_on: str = "complex",
    ) -> None:
        super().__init__()
        enc_layer = TransformerEncoderLayer(
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            layer_norm_eps,
            batch_first,
            softmax_on,
        )
        self.encoder = TransformerEncoder(enc_layer, num_encoder_layers)
        dec_layer = TransformerDecoderLayer(
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            layer_norm_eps,
            batch_first,
            softmax_on,
        )
        self.decoder = TransformerDecoder(dec_layer, num_decoder_layers)
        self.d_model = d_model
        self.nhead = nhead
        self.batch_first = batch_first

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        memory = self.encoder(src)
        return self.decoder(tgt, memory)
