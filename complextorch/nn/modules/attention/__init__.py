import torch
import torch.nn as nn

from complextorch import nn as cvnn
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
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        r"""Implements the complex-valued scaled dot-product attention operation.

        Args:
            q (torch.Tensor): complex-valued query tensor
            k (torch.Tensor): complex-valued key tensor
            v (torch.Tensor): complex-valued value tensor

        Returns:
            torch.Tensor: \mathcal{S}(Q K^H / t) V
        """
        # Conjugate transpose so the dot product is the Hermitian inner product Q K^H.
        attn = torch.matmul(q / self.temperature, k.conj().transpose(-2, -1))

        if self.softmax_on == "real":
            # Softmax on Re[QK^H]: real-valued attention weights × complex V.
            weights = self.softmax(attn.real)
            weights = self.dropout(weights.to(v.dtype))
            return torch.matmul(weights, v)

        attn = self.dropout(self.softmax(attn))
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
        self.layer_norm = cvnn.LayerNorm(d_model, eps=1e-6)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        d_k, d_v, n_heads = self.d_k, self.d_v, self.n_heads
        batch_size, len_q, len_k, len_v = q.shape[0], q.shape[1], k.shape[1], v.shape[1]

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

        q = self.attention(q, k, v)

        q = q.transpose(1, 2).contiguous().view(batch_size, len_q, -1)
        q = self.dropout(self.fc(q))
        q += res

        return self.layer_norm(q)
