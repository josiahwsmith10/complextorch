import numpy as np
import torch
import torch.nn as nn

from .... import CVTensor
from .... import nn as cvnn

__all__ = [
    'CVScaledDotProductAttention',
    'CVMultiheadAttention'
]


class CVScaledDotProductAttention(nn.Module):
    """Complex-Valued Scaled Dot-Product Attention."""

    def __init__(
        self,
        temperature: float,
        attn_dropout: float = 0.1,
        SoftMaxClass: nn.Module = cvnn.MagMinMaxNorm,
    ) -> None:
        super(CVScaledDotProductAttention, self).__init__()

        self.temperature = temperature
        self.dropout = cvnn.CVDropout(attn_dropout)
        self.softmax = SoftMaxClass(dim=-1)

    def forward(self, q: CVTensor, k: CVTensor, v: CVTensor) -> CVTensor:
        attn = torch.matmul(q.complex / self.temperature, k.complex.transpose(-2, -1))

        attn = self.dropout(self.softmax(attn))
        output = torch.matmul(attn.complex, v.complex)
        return CVTensor(output.real, output.imag)


class CVMultiheadAttention(nn.Module):
    def __init__(
        self,
        n_heads: int,
        d_model: int,
        d_k: int,
        d_v: int,
        dropout: float = 0.1,
        SoftMaxClass: nn.Module = cvnn.MagMinMaxNorm,
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


class CVEfficientChannelAttention1d(nn.Module):
    """Complex-valued efficient channel attention."""

    def __init__(self, channels: int, b: int = 1, gamma: int = 2) -> None:
        super(CVEfficientChannelAttention1d, self).__init__()
        self.avg_pool = cvnn.CVAdaptiveAvgPool1d(1)
        self.channels = channels
        self.b = b
        self.gamma = gamma
        self.conv = cvnn.CVConv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=self.kernel_size(),
            padding=(self.kernel_size() - 1) // 2,
            bias=False,
        )
        self.sigmoid = cvnn.CSigmoid()

    def kernel_size(self) -> int:
        k = int(abs((np.log2(self.channels) / self.gamma) + self.b / self.gamma))
        out = k if k % 2 else k + 1
        return out

    def forward(self, x: CVTensor) -> CVTensor:
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.transpose(-1, -2)).transpose(-1, -2)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)

class CVMaskedChannelAttention1d(nn.Module):
    """
    Complex-Valued Masked Channel Attention Module.

    HW Cho, S Choi, YR Cho, and J Kim: Complex-Valued Channel Attention and Application in Ego-Velocity Estimation With Automotive Radar
    Fig. 3
    https://ieeexplore.ieee.org/abstract/document/9335579
    
    Generalized for arbitrary masking function (see mask.py for implemented masking functions)
    """

    def __init__(
        self, 
        channels: int, 
        reduction_factor: int = 2,
        MaskingClass: nn.Module = cvnn.ComplexRatioMask,
        act: nn.Module = cvnn.CReLU,
    ) -> None:
        super(CVMaskedChannelAttention1d, self).__init__()
        self.channels = channels
        self.reduction_factor = reduction_factor
        self.MaskingClass = MaskingClass()
        self.act = act()
        
        self.avg_pool = cvnn.CVAdaptiveAvgPool1d(1)
        
        assert channels % reduction_factor == 0, "Channels / Reduction Factor must yield integer"
        
        reduced_channels = int(channels / reduction_factor)
        
        self.conv_down = cvnn.CVConv1d(
            in_channels=channels,
            out_channels=reduced_channels,
            kernel_size=1,
            bias=False,
        )
        
        self.conv_up = cvnn.CVConv1d(
            in_channels=reduced_channels,
            out_channels=channels,
            kernel_size=1,
            bias=False,
        )
        
    def forward(self, x: CVTensor) -> CVTensor:
        # Get attention values
        attn = self.conv_up(self.act(self.conv_down(x)))
        
        # Compute mask
        mask = self.MaskingClass(attn)
        
        return x * mask
