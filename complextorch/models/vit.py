r"""
Complex-Valued Vision Transformer (ViT)
=======================================

Pre-built complex-valued ViT with the standard `t/s/b/l/h` size presets,
mirroring :mod:`torchvision.models.vit` and ``torchcvnn.models.vit``.

Inputs are complex images of shape ``(B, in_channels, H, W)``. Patch
embedding is performed with a single complex :class:`Conv2d` whose kernel
and stride both equal ``patch_size``.
"""

import torch
import torch.nn as nn

from complextorch.nn.modules.activation.split_type_A import CGELU
from complextorch.nn.modules.attention import MultiheadAttention
from complextorch.nn.modules.conv import Conv2d
from complextorch.nn.modules.dropout import Dropout
from complextorch.nn.modules.layernorm import LayerNorm
from complextorch.nn.modules.linear import Linear
from complextorch.nn.modules.position import (
    RotaryEmbedding,
    SinusoidalPositionalEncoding,
)

__all__ = ["ViT", "ViTLayer", "vit_b", "vit_h", "vit_l", "vit_s", "vit_t"]


class _ViTFFN(nn.Module):
    """Pre-norm MLP block with residual."""

    def __init__(self, dim: int, mlp_dim: int, dropout: float, eps: float) -> None:
        super().__init__()
        self.norm = LayerNorm(dim, eps=eps)
        self.fc1 = Linear(dim, mlp_dim)
        self.act = CGELU()
        self.fc2 = Linear(mlp_dim, dim)
        self.drop = Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.drop(self.fc2(self.act(self.fc1(self.norm(x)))))


class ViTLayer(nn.Module):
    r"""
    Single Vision Transformer block (pre-norm).

    Composition: pre-norm :class:`LayerNorm` -> :class:`MultiheadAttention`
    (which itself adds residual + LayerNorm internally) -> pre-norm FFN with
    its own residual.
    """

    def __init__(
        self,
        dim: int,
        nhead: int,
        mlp_dim: int,
        dropout: float = 0.0,
        layer_norm_eps: float = 1e-5,
        softmax_on: str = "complex",
        rotary: RotaryEmbedding | None = None,
    ) -> None:
        super().__init__()
        if dim % nhead != 0:
            raise ValueError(f"dim ({dim}) must be divisible by nhead ({nhead})")
        d_head = dim // nhead
        self.attn = MultiheadAttention(
            nhead,
            dim,
            d_head,
            d_head,
            dropout=dropout,
            softmax_on=softmax_on,
            rotary=rotary,
        )
        self.ffn = _ViTFFN(dim, mlp_dim, dropout, layer_norm_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.attn(x, x, x)
        return self.ffn(x)


class ViT(nn.Module):
    r"""
    Complex-Valued Vision Transformer.

    Args:
        image_size: input spatial size (assumed square).
        patch_size: patch side length.
        in_channels: number of input channels (complex).
        num_classes: classifier dimensionality (set to 0 to disable the head).
        dim: embedding dim.
        depth: number of :class:`ViTLayer` blocks.
        heads: number of attention heads.
        mlp_dim: width of the FFN.
        dropout: dropout probability.
        softmax_on: ``'complex'`` or ``'real'``; controls attention softmax.
        pos_encoding: positional-encoding scheme:

            - ``'learned'`` (default): a learned complex ``pos_embed`` parameter
              added to the token + patch embeddings (original behaviour).
            - ``'sinusoidal'``: a fixed complex
              :class:`~complextorch.nn.SinusoidalPositionalEncoding` added to the
              embeddings.
            - ``'rotary'``: relative :class:`~complextorch.nn.RotaryEmbedding`
              applied to the per-head query/key tensors inside attention; no
              additive position embedding is used.

    Note: the classification head returns a complex ``(B, num_classes)``
    tensor. Most downstream losses take ``|·|`` first.
    """

    def __init__(
        self,
        image_size: int,
        patch_size: int,
        in_channels: int = 1,
        num_classes: int = 1000,
        dim: int = 768,
        depth: int = 12,
        heads: int = 12,
        mlp_dim: int = 3072,
        dropout: float = 0.0,
        layer_norm_eps: float = 1e-5,
        softmax_on: str = "complex",
        pos_encoding: str = "learned",
    ) -> None:
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError(
                f"image_size ({image_size}) must be divisible by patch_size ({patch_size})"
            )
        if pos_encoding not in ("learned", "sinusoidal", "rotary"):
            raise ValueError(
                "pos_encoding must be 'learned', 'sinusoidal' or 'rotary'; "
                f"got {pos_encoding!r}"
            )
        if dim % heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by heads ({heads})")
        num_patches = (image_size // patch_size) ** 2
        self.pos_encoding = pos_encoding
        self.patch_embed = Conv2d(
            in_channels, dim, kernel_size=patch_size, stride=patch_size, bias=True
        )
        # Class token is a complex parameter.
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim, dtype=torch.cfloat))
        with torch.no_grad():
            self.cls_token.real.normal_(0, 0.02)
            self.cls_token.imag.normal_(0, 0.02)

        self.pos_embed: nn.Parameter | None = None
        self.pos_enc: SinusoidalPositionalEncoding | None = None
        rotary: RotaryEmbedding | None = None
        if pos_encoding == "learned":
            self.pos_embed = nn.Parameter(
                torch.zeros(1, num_patches + 1, dim, dtype=torch.cfloat)
            )
            with torch.no_grad():
                self.pos_embed.real.normal_(0, 0.02)
                self.pos_embed.imag.normal_(0, 0.02)
        elif pos_encoding == "sinusoidal":
            self.pos_enc = SinusoidalPositionalEncoding(dim)
        else:  # rotary
            rotary = RotaryEmbedding(dim // heads)

        self.drop = Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                ViTLayer(
                    dim, heads, mlp_dim, dropout, layer_norm_eps, softmax_on, rotary
                )
                for _ in range(depth)
            ]
        )
        self.norm = LayerNorm(dim, eps=layer_norm_eps)
        self.head = Linear(dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        x = self.patch_embed(x)  # (B, dim, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)  # (B, N, dim)
        cls = self.cls_token.expand(b, -1, -1)
        x = torch.cat([cls, x], dim=1)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        elif self.pos_enc is not None:
            x = self.pos_enc(x)
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        cls_out = x[:, 0]
        return self.head(cls_out)


# ---------------------------------------------------------------------------
# Size presets (match the standard ViT family).
# ---------------------------------------------------------------------------


def vit_t(image_size: int = 224, patch_size: int = 16, **kwargs) -> ViT:
    """ViT-Tiny: 12 layers, 3 heads, 192 dim."""
    return ViT(
        image_size=image_size,
        patch_size=patch_size,
        dim=192,
        depth=12,
        heads=3,
        mlp_dim=768,
        **kwargs,
    )


def vit_s(image_size: int = 224, patch_size: int = 16, **kwargs) -> ViT:
    """ViT-Small: 12 layers, 6 heads, 384 dim."""
    return ViT(
        image_size=image_size,
        patch_size=patch_size,
        dim=384,
        depth=12,
        heads=6,
        mlp_dim=1536,
        **kwargs,
    )


def vit_b(image_size: int = 224, patch_size: int = 16, **kwargs) -> ViT:
    """ViT-Base: 12 layers, 12 heads, 768 dim."""
    return ViT(
        image_size=image_size,
        patch_size=patch_size,
        dim=768,
        depth=12,
        heads=12,
        mlp_dim=3072,
        **kwargs,
    )


def vit_l(image_size: int = 224, patch_size: int = 16, **kwargs) -> ViT:
    """ViT-Large: 24 layers, 16 heads, 1024 dim."""
    return ViT(
        image_size=image_size,
        patch_size=patch_size,
        dim=1024,
        depth=24,
        heads=16,
        mlp_dim=4096,
        **kwargs,
    )


def vit_h(image_size: int = 224, patch_size: int = 14, **kwargs) -> ViT:
    """ViT-Huge: 32 layers, 16 heads, 1280 dim."""
    return ViT(
        image_size=image_size,
        patch_size=patch_size,
        dim=1280,
        depth=32,
        heads=16,
        mlp_dim=5120,
        **kwargs,
    )
