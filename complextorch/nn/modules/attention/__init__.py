import torch
import torch.nn as nn

from complextorch import nn as cvnn
from complextorch.nn.modules.attention._mask import _canonical_mask
from complextorch.nn.modules.attention.holographic import HolographicAttention
from complextorch.nn.modules.position import RotaryEmbedding

__all__ = ["MultiheadAttention", "ScaledDotProductAttention"]


class ScaledDotProductAttention(nn.Module):
    r"""
    Complex-Valued Scaled Dot-Product Attention
    -------------------------------------------

    The ever-popular scaled dot-product attention is the backbone of many attention-based methods, most notably the transformer.

    Implements the operation:

    .. math::

        \text{Attention}(Q, K, V) = \mathcal{S}(Q K^T / t) V

    where :math:`Q, K, V` are complex-valued tensors, :math:`t` is known as the temperature typically :math:`t = \sqrt{d_{attn}}`, and :math:`\mathcal{S}` is the softmax function.

    For complex-values, the `traditional softmax function <https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html>`_ cannot be applied, and variants must be applied.
    Included in this library are several options for :mod:`complex-valued softmax <complextorch.nn.modules.softmax>` and similar :mod:`masking <complextorch.nn.modules.mask>` functions.

    By default, the :class:`ScaledDotProductAttention` employs :class:`complextorch.nn.CVSoftMax`, which applies the traditional softmax to the real and imaginary parts of the complex-valued tensor separately. If a phase-preserving alternative is preferred, pass :class:`complextorch.nn.PhaseSoftMax` via the ``SoftMaxClass`` argument.

    .. note::

        When ``attn_mask`` is used with ``softmax_on='complex'``, the mask is
        routed *into* the softmax module so each variant can apply it to the
        real statistic it exponentiates. A custom ``SoftMaxClass`` must
        therefore accept ``(input, mask)`` to be used with masks;
        single-argument classes keep working unmasked.
    """

    def __init__(
        self,
        temperature: float,
        attn_dropout: float = 0.1,
        SoftMaxClass: nn.Module = cvnn.CVSoftMax,
        softmax_on: str = "complex",
    ) -> None:
        super().__init__()
        if softmax_on not in ("complex", "real"):
            raise ValueError(
                f"softmax_on must be 'complex' or 'real', got {softmax_on!r}"
            )

        self.temperature = temperature
        self.dropout = cvnn.Dropout(attn_dropout)
        self.softmax_on = softmax_on
        if softmax_on == "complex":
            self.softmax = SoftMaxClass(dim=-1)
        else:
            self.softmax = nn.Softmax(dim=-1)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        r"""Implements the complex-valued scaled dot-product attention operation.

        Args:
            q (torch.Tensor): complex-valued query tensor
            k (torch.Tensor): complex-valued key tensor
            v (torch.Tensor): complex-valued value tensor
            attn_mask (torch.Tensor, optional): bool (``True`` = disallowed)
                or real additive float mask (``-inf`` = disallowed),
                broadcastable to the score shape ``(..., L_q, L_k)``. Masked
                positions receive exactly zero attention weight. A fully
                masked row yields NaN, matching ``torch`` softmax semantics.

        Returns:
            torch.Tensor: \mathcal{S}(Q K^H / t) V
        """
        attn_mask = _canonical_mask(attn_mask, "attn_mask")
        # Conjugate transpose so the dot product is the Hermitian inner product Q K^H.
        attn = torch.matmul(q / self.temperature, k.conj().transpose(-2, -1))

        if self.softmax_on == "real":
            # Softmax on Re[QK^H]: real-valued attention weights × complex V.
            logits = attn.real if attn_mask is None else attn.real + attn_mask
            weights = self.softmax(logits)
            weights = self.dropout(weights.to(v.dtype))
            return torch.matmul(weights, v)

        # Complex mode: route the mask through the softmax variant so masked
        # positions get exactly zero weight under split *and* polar softmax.
        attn = (
            self.softmax(attn) if attn_mask is None else self.softmax(attn, attn_mask)
        )
        attn = self.dropout(attn)
        return torch.matmul(attn, v)


class MultiheadAttention(nn.Module):
    r"""
    Complex-Valued Multihead Attention
    ----------------------------------

    Multihead self attention extended to complex-valued tensors.

    By default, the :class:`MultiheadAttention` employs :class:`complextorch.nn.CVSoftMax`, which applies the traditional softmax to the real and imaginary parts of the complex-valued tensor separately. Pass :class:`complextorch.nn.PhaseSoftMax` via ``SoftMaxClass`` for the phase-preserving alternative.

    Pass a :class:`complextorch.nn.RotaryEmbedding` via ``rotary`` to inject
    relative positional information; it is applied to the per-head query and key
    tensors after projection and must be constructed with ``dim=d_k``. Rotary is
    intended for **self-attention** (a shared query/key position axis); for
    cross-attention with differing query/key positions the relative-offset
    semantics are not meaningful, so prefer an additive encoding there.

    Set ``attention='holographic'`` to use the interference-aware
    :class:`~complextorch.nn.HolographicAttention` core instead of the default
    scaled dot-product attention (in which case ``SoftMaxClass`` / ``softmax_on``
    are unused).

    By default this module is a *complete* post-norm attention sub-block: the
    projected attention output is followed by an internal residual connection
    and :class:`LayerNorm`. Pass ``residual_norm=False`` to get only the
    projected attention output, for callers that supply their own residual
    and normalization (e.g. pre-norm blocks like :class:`ViTLayer`).
    """

    def __init__(
        self,
        n_heads: int,
        d_model: int,
        d_k: int,
        d_v: int,
        dropout: float = 0.1,
        SoftMaxClass: nn.Module = cvnn.CVSoftMax,
        softmax_on: str = "complex",
        rotary: RotaryEmbedding | None = None,
        attention: str = "scaled_dot_product",
        residual_norm: bool = True,
    ) -> None:
        super().__init__()
        if attention not in ("scaled_dot_product", "holographic"):
            raise ValueError(
                "attention must be 'scaled_dot_product' or 'holographic'; "
                f"got {attention!r}"
            )

        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        self.rotary = rotary
        self.residual_norm = residual_norm

        self.w_q = cvnn.Linear(d_model, n_heads * d_k, bias=False)
        self.w_k = cvnn.Linear(d_model, n_heads * d_k, bias=False)
        self.w_v = cvnn.Linear(d_model, n_heads * d_v, bias=False)
        self.fc = cvnn.Linear(n_heads * d_v, d_model, bias=False)

        if attention == "holographic":
            self.attention = HolographicAttention(
                temperature=d_k**0.5, attn_dropout=dropout
            )
        else:
            self.attention = ScaledDotProductAttention(
                temperature=d_k**0.5,
                attn_dropout=dropout,
                SoftMaxClass=SoftMaxClass,
                softmax_on=softmax_on,
            )

        self.dropout = cvnn.Dropout(dropout)
        if residual_norm:
            self.layer_norm = cvnn.LayerNorm(d_model, eps=1e-6)
        else:
            self.layer_norm = None

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        r"""Computes complex-valued multihead attention.

        Args:
            q (torch.Tensor): complex-valued query tensor ``(B, L_q, d_model)``
            k (torch.Tensor): complex-valued key tensor ``(B, L_k, d_model)``
            v (torch.Tensor): complex-valued value tensor ``(B, L_k, d_model)``
            attn_mask (torch.Tensor, optional): bool (``True`` = disallowed)
                or additive float mask of shape ``(L_q, L_k)`` or
                broadcastable to ``(B, n_heads, L_q, L_k)``.
            key_padding_mask (torch.Tensor, optional): bool (``True`` = pad)
                or additive float mask of shape ``(B, L_k)``.

        Returns:
            torch.Tensor: attended tensor ``(B, L_q, d_model)``
        """
        d_k, d_v, n_heads = self.d_k, self.d_v, self.n_heads
        batch_size, len_q, len_k, len_v = q.shape[0], q.shape[1], k.shape[1], v.shape[1]

        mask = _canonical_mask(attn_mask, "attn_mask")
        kpm = _canonical_mask(key_padding_mask, "key_padding_mask")
        if kpm is not None:
            if kpm.shape != (batch_size, len_k):
                raise ValueError(
                    f"key_padding_mask must have shape ({batch_size}, {len_k}), "
                    f"got {tuple(kpm.shape)}"
                )
            # Broadcast over heads and query positions; merge by addition
            # (-inf + anything = -inf), matching torch's mask merge.
            kpm = kpm.view(batch_size, 1, 1, len_k)
            mask = kpm if mask is None else mask + kpm

        res = q

        q = self.w_q(q).view(batch_size, len_q, n_heads, d_k)
        k = self.w_k(k).view(batch_size, len_k, n_heads, d_k)
        v = self.w_v(v).view(batch_size, len_v, n_heads, d_v)

        (
            q,
            k,
            v,
        ) = (
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
        )

        if self.rotary is not None:
            q, k = self.rotary.rotate_q_k(q, k)

        q = self.attention(q, k, v, attn_mask=mask)

        q = q.transpose(1, 2).contiguous().view(batch_size, len_q, -1)
        q = self.dropout(self.fc(q))
        if not self.residual_norm:
            return q
        q += res

        return self.layer_norm(q)
