import math

import torch
import torch.nn as nn

from complextorch.nn.modules.activation.split_type_A import GeneralizedSplitActivation

__all__ = [
    "CPReLU",
    "CReLU",
    "CVSplitReLU",
    "EquivariantPhaseReLU",
    "GTReLU",
    "zAbsReLU",
    "zLeakyReLU",
]


class CVSplitReLU(GeneralizedSplitActivation):
    r"""
    Split Complex-Valued Rectified Linear Unit
    ------------------------------------------

    Implements the operation:

    .. math::

        G(\mathbf{z}) = \texttt{ReLU}(\mathbf{x}) + j \texttt{ReLU}(\mathbf{y})

    Based on work from the following paper:

        **Jingkun Gao, Bin Deng, Yuliang Qin, Hongqiang Wang and Xiang Li. Enhanced Radar Imaging Using a Complex-valued Convolutional Neural Network.**

            - Eq. (5)

            - https://arxiv.org/abs/1712.10096
    """

    def __init__(self, inplace: bool = True) -> None:
        super().__init__(nn.ReLU(inplace), nn.ReLU(inplace))


class CReLU(CVSplitReLU):
    r"""
    Split Complex-Valued Rectified Linear Unit
    ------------------------------------------

    Implements the operation:

    .. math::

        G(\mathbf{z}) = \texttt{ReLU}(\mathbf{x}) + j \texttt{ReLU}(\mathbf{y})

    Alias for :class:`CVSplitReLU`. The nomenclature CReLU is used only in certain literature to denote the split complex-valued rectified linear unit.
    """


class CPReLU(GeneralizedSplitActivation):
    r"""
    Split Complex-Valued Parametric Rectified Linear Unit
    -----------------------------------------------------

    Split Type-A extension of the `Parametric ReLU <https://pytorch.org/docs/stable/generated/torch.nn.PReLU.html>`_ for complex-valued tensors.

    Implements the operation:

    .. math::

        G(\mathbf{z}) = \texttt{PReLU}(\mathbf{x}) + j \texttt{PReLU}(\mathbf{y})

    Based on work from the following paper:

        **H. Jing, S. Li, K. Miao, S. Wang, X. Cui, G. Zhao and H. Sun. Enhanced Millimeter-Wave 3-D Imaging via Complex-Valued Fully Convolutional Neural Network.**

            - Eq. (2)

            - https://www.mdpi.com/2079-9292/11/1/147
    """

    def __init__(self) -> None:
        super().__init__(nn.PReLU(), nn.PReLU())


class zAbsReLU(nn.Module):
    r"""
    Magnitude-Thresholded ReLU with Learnable Threshold
    ---------------------------------------------------

    Zeros out elements whose magnitude is below a learnable threshold
    :math:`a`, preserving the phase of passing elements:

    .. math::

        \texttt{zAbsReLU}(z) = \begin{cases}
            z & \text{if } |z| \geq a \\
            0 & \text{otherwise}
        \end{cases}

    Args:
        a_init: initial value of the (scalar) threshold parameter.
    """

    def __init__(self, a_init: float = 0.0) -> None:
        super().__init__()
        self.a = nn.Parameter(torch.tensor(float(a_init)))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        mask = (input.abs() >= self.a).to(
            input.dtype if input.is_complex() else input.real.dtype
        )
        if input.is_complex():
            return input * mask
        return input * mask


class zLeakyReLU(nn.Module):
    r"""
    Leaky First-Quadrant Complex ReLU
    ---------------------------------

    Soft version of :class:`zReLU`: passes :math:`z` unchanged when both
    :math:`\Re z > 0` and :math:`\Im z > 0`; scales by ``negative_slope``
    elsewhere.

    .. math::

        \texttt{zLeakyReLU}(z) = \begin{cases}
            z & \text{if } \Re z > 0 \text{ and } \Im z > 0 \\
            \alpha\, z & \text{otherwise}
        \end{cases}
    """

    def __init__(self, negative_slope: float = 0.01) -> None:
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        in_q1 = (input.real > 0) & (input.imag > 0)
        scale = torch.where(
            in_q1,
            torch.ones_like(input.real),
            torch.full_like(input.real, self.negative_slope),
        )
        return input * scale

    def extra_repr(self) -> str:
        return f"negative_slope={self.negative_slope}"


# ---------------------------------------------------------------------------
# CDS phase-thresholding activations (Singhal, Xing, Yu — CVPR 2022)
# ---------------------------------------------------------------------------


class _PhaseHalfPlaneMask(torch.autograd.Function):
    r"""Forward: :math:`\theta \mapsto \theta \cdot \mathbf{1}[\theta \bmod 2\pi \in [0, \pi]]`.
    Backward: the gradient is the mask itself.

    Matches ``Two_Channel_Nonlinearity`` from ``cds/layers.py:494-520``.
    """

    @staticmethod
    def forward(ctx, phase: torch.Tensor) -> torch.Tensor:
        wrapped = phase % (2.0 * math.pi)
        mask = ((wrapped >= 0.0) & (wrapped <= math.pi)).to(phase.dtype)
        ctx.save_for_backward(mask)
        return phase * mask

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (mask,) = ctx.saved_tensors
        return grad_output * mask


def _broadcast_channelwise(t: torch.Tensor, input_dim: int) -> torch.Tensor:
    """Broadcast a 1-D parameter ``(C,)`` to the channel dim (1) of an N-D input."""
    if t.dim() == 1 and input_dim > 1:
        shape = [1] * input_dim
        shape[1] = t.shape[0]
        return t.view(*shape)
    return t


class GTReLU(nn.Module):
    r"""
    Gabor-Tangent ReLU (CDS)
    ------------------------

    Phase-thresholding nonlinearity used in the I-type CDS network. Composes a
    learnable complex scaling, a half-plane phase mask, and an optional
    learnable phase rescaling:

    1. Scale: :math:`z' = (\alpha + j\beta) \cdot z`, with per-channel
       :math:`\alpha, \beta \in \mathbb{R}^C` learnable.
    2. Mask phase: pass only phases in the upper half-plane:

       .. math::

           \tilde{\theta} = \arg(z') \cdot \mathbf{1}[\arg(z') \bmod 2\pi \in [0, \pi]]

    3. Recombine: :math:`\operatorname{out} = |z'| \cdot e^{j\tilde{\theta}}`.
    4. (Optional, when ``phase_scale=True``) rescale the masked phase by
       :math:`\operatorname{clamp}(\lambda, 0.5, 2)` with :math:`\lambda \in \mathbb{R}^C`
       learnable (initialised to 1).

    The mask gradient is implemented via a custom
    :class:`torch.autograd.Function` (the mask itself is the gradient).

    Based on work from the following paper:

        **U. Singhal, Y. Xing, S. X. Yu. Co-Domain Symmetry for Complex-Valued Deep Learning.**

            - CVPR 2022 — `GTReLU` in the reference implementation

            - https://openaccess.thecvf.com/content/CVPR2022/papers/Singhal_Co-Domain_Symmetry_for_Complex-Valued_Deep_Learning_CVPR_2022_paper.pdf

    Args:
        num_channels: number of complex channels.
        global_scaling: if True, share a single scalar :math:`(\alpha, \beta)`
            across all channels.
        phase_scale: if True, add the per-channel learnable phase rescale.
    """

    def __init__(
        self,
        num_channels: int,
        global_scaling: bool = False,
        phase_scale: bool = False,
    ) -> None:
        super().__init__()
        n = 1 if global_scaling else num_channels
        self.num_channels = num_channels
        self.global_scaling = global_scaling
        self.phase_scale = phase_scale
        self.alpha = nn.Parameter(torch.empty(n).uniform_(0.0, 1.0))
        self.beta = nn.Parameter(torch.empty(n).uniform_(0.0, 1.0))
        if phase_scale:
            self.lambd = nn.Parameter(torch.ones(num_channels))
        else:
            self.register_parameter("lambd", None)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if not input.is_complex():
            input = input.to(torch.cfloat)
        alpha = _broadcast_channelwise(self.alpha, input.dim())
        beta = _broadcast_channelwise(self.beta, input.dim())
        scale = torch.complex(alpha, beta)
        z = input * scale
        magnitude = z.abs()
        phase = z.angle()
        masked_phase = _PhaseHalfPlaneMask.apply(phase)
        if self.lambd is not None:
            lam = _broadcast_channelwise(self.lambd, input.dim())
            masked_phase = masked_phase * lam.clamp(min=0.5, max=2.0)
        return torch.polar(magnitude, masked_phase)

    def extra_repr(self) -> str:
        return (
            f"num_channels={self.num_channels}, global_scaling={self.global_scaling}, "
            f"phase_scale={self.phase_scale}"
        )


class EquivariantPhaseReLU(nn.Module):
    r"""
    Equivariant Phase ReLU (CDS)
    ----------------------------

    The U(1)-equivariant counterpart of :class:`GTReLU`. Thresholds phase
    *relative to the channel-mean direction*, so the operator commutes with
    any global phase rotation:

    1. Channel-mean reference direction:

       .. math::

           \hat{p} = \frac{\mathrm{mean}_c(z)}{|\mathrm{mean}_c(z)| + \varepsilon}

    2. Relative phase: :math:`\varphi = \arg(z \cdot \overline{\hat{p}})`.
    3. Threshold (half-plane mask, then per-channel scale):

       .. math::

           \tilde{\varphi} = \varphi \cdot \mathbf{1}[\varphi \bmod 2\pi \in [0, \pi]] \cdot \operatorname{ReLU}(s)

       with :math:`s \in \mathbb{R}^C` learnable (init 1).
    4. Output: :math:`|z| \cdot e^{j\tilde{\varphi}} \cdot \hat{p}`.

    Rotating the input by :math:`e^{j\psi}` rotates :math:`\hat{p}` by the same
    angle, leaving :math:`\varphi` invariant and rotating the output by
    :math:`\psi` — exact U(1)-equivariance.

    Based on work from the following paper:

        **U. Singhal, Y. Xing, S. X. Yu. Co-Domain Symmetry for Complex-Valued Deep Learning.**

            - CVPR 2022 — `eqnl` in the reference implementation

            - https://openaccess.thecvf.com/content/CVPR2022/papers/Singhal_Co-Domain_Symmetry_for_Complex-Valued_Deep_Learning_CVPR_2022_paper.pdf

    Args:
        num_channels: number of complex channels in the input.
        eps: numerical floor when normalising the channel mean.
    """

    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.eps = eps
        self.phase_gain = nn.Parameter(torch.ones(num_channels))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if not input.is_complex():
            input = input.to(torch.cfloat)
        # Channel-mean reference direction (keepdim so it broadcasts back).
        ref = input.mean(dim=1, keepdim=True)
        ref = ref / (ref.abs() + self.eps)
        # Relative phase: arg(z * conj(ref))
        relative_phase = (input * ref.conj()).angle()
        masked = _PhaseHalfPlaneMask.apply(relative_phase)
        gain = _broadcast_channelwise(self.phase_gain, input.dim())
        masked = masked * torch.relu(gain)
        return input.abs() * torch.polar(torch.ones_like(masked), masked) * ref

    def extra_repr(self) -> str:
        return f"num_channels={self.num_channels}, eps={self.eps}"
