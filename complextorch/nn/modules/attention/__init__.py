import numpy as np
import torch
import torch.nn as nn

from .... import CVTensor
from .... import nn as cvnn

__all__ = ["CVScaledDotProductAttention", "CVMultiheadAttention"]


class CVScaledDotProductAttention(nn.Module):
    r"""
    Complex-Valued Scaled Dot-Product Attention
    -------------------------------------------

    The ever-popular scaled dot-product attention is the backbone of many attention-based methods, most notably the transformer.

    Implements the operation:

    .. math::

        G(Q, K, V) = \mathcal{S}(Q K^T / t) V

    where :math:`Q, K, V` are complex-valued tensors, :math:`t` is known as the temperature typically :math:`t = \sqrt{d_{attn}}`, and :math:`\mathcal{S}` is the softmax function.

    For complex-values, the `traditional softmax function <https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html>`_ cannot be applied, and variants must be applied.
    Included in this library are several options for :doc:`complex-valued softmax <./softmax>` and similar :doc:`masking <./mask>` functions.

    By default, the :class:`CVScaledDotProductAttention` employs the :class:`complextorch.nn.CVSoftmax`, which applies the traditional softmax to the magnitude of the complex-valued tensor while leaving the phase information unchanged.
    """

    def __init__(
        self,
        temperature: float,
        attn_dropout: float = 0.1,
        SoftMaxClass: nn.Module = cvnn.CVSoftMax,
    ) -> None:
        super(CVScaledDotProductAttention, self).__init__()

        self.temperature = temperature
        self.dropout = cvnn.CVDropout(attn_dropout)
        self.softmax = SoftMaxClass(dim=-1)

    def forward(self, q: CVTensor, k: CVTensor, v: CVTensor) -> CVTensor:
        r"""Implements the complex-valued scaled dot-product attention operation.

        Args:
            q (CVTensor): complex-valued query tensor
            k (CVTensor): complex-valued key tensor
            v (CVTensor): complex-valued value tensor

        Returns:
            CVTensor: \mathcal{S}(Q K^T / t) V
        """
        attn = torch.matmul(q.complex / self.temperature, k.complex.transpose(-2, -1))

        attn = self.dropout(self.softmax(attn))
        output = torch.matmul(attn.complex, v.complex)
        return CVTensor(output.real, output.imag)


class CVMultiheadAttention(nn.Module):
    r"""
    Complex-Valued Multihead Attention
    ----------------------------------

    Multihead self attention extended to complex-valued tensors.

    By default, the :class:`CVMultiheadAttention` employs the :class:`complextorch.nn.CVSoftmax`, which applies the traditional softmax to the magnitude of the complex-valued tensor while leaving the phase information unchanged.
    """

    def __init__(
        self,
        n_heads: int,
        d_model: int,
        d_k: int,
        d_v: int,
        dropout: float = 0.1,
        SoftMaxClass: nn.Module = cvnn.CVSoftMax,
    ) -> None:
        super(CVMultiheadAttention, self).__init__()

        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads

        self.w_q = cvnn.CVLinear(d_model, n_heads * d_k, bias=False)
        self.w_k = cvnn.CVLinear(d_model, n_heads * d_k, bias=False)
        self.w_v = cvnn.CVLinear(d_model, n_heads * d_v, bias=False)
        self.fc = cvnn.CVLinear(n_heads * d_v, d_model, bias=False)

        self.attention = CVScaledDotProductAttention(
            temperature=d_k**0.5, attn_dropout=dropout, SoftMaxClass=SoftMaxClass
        )

        self.dropout = cvnn.CVDropout(dropout)
        self.layer_norm = cvnn.CVLayerNorm(d_model, eps=1e-6)

    def forward(self, q: CVTensor, k: CVTensor, v: CVTensor) -> CVTensor:
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

        q = self.attention(q, k, v)

        q = q.transpose(1, 2).contiguous().view(batch_size, len_q, -1)
        q = self.dropout(self.fc(q))
        q += res

        return self.layer_norm(q)
