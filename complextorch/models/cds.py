r"""
CDS (Co-Domain Symmetry) Reference Architectures
================================================

Three reference networks from Singhal, Xing, Yu — *Co-Domain Symmetry for
Complex-Valued Deep Learning* (CVPR 2022):

- :class:`CDSInvariant` (`I`-type) — uses :class:`PhaseDivConv` to make the
  representation invariant to a global phase rotation of the input.
- :class:`CDSEquivariant` (`E`-type) — uses :class:`ComplexScaling` +
  :class:`EquivariantPhaseReLU` + :class:`MagBatchNorm2d` for full
  U(1)-equivariance, with a phase-rotated prototype head producing invariant
  logits.
- :class:`CDSMSTAR` — SAR-style backbone using :class:`PhaseConjConv` and a
  real-valued ResNet-lite tail.

Reference: https://openaccess.thecvf.com/content/CVPR2022/papers/Singhal_Co-Domain_Symmetry_for_Complex-Valued_Deep_Learning_CVPR_2022_paper.pdf
"""

import math

import torch
import torch.nn as nn

from complextorch.nn import (
    ComplexScaling,
    Conv2d,
    EquivariantPhaseReLU,
    GTReLU,
    MagBatchNorm2d,
    MagMaxPool2d,
    PhaseConjConv2d,
    PhaseDivConv2d,
    PrototypeDistance,
)

__all__ = ["CDSInvariant", "CDSEquivariant", "CDSMSTAR"]


def _complex_to_real_flat(z: torch.Tensor) -> torch.Tensor:
    """Stack real and imag parts on the channel dim: [B, C, ...] → [B, 2C, ...]."""
    return torch.cat([z.real, z.imag], dim=1)


def _real_flat_to_complex(r: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`_complex_to_real_flat`: [B, 2C, ...] → [B, C, ...] cfloat."""
    c = r.shape[1] // 2
    return torch.complex(r[:, :c], r[:, c:])


class CDSInvariant(nn.Module):
    r"""
    CDS Invariant (I-Type) Network
    ------------------------------

    Small CDS variant for CIFAR-style experiments. The :class:`PhaseDivConv2d`
    after the first convolution makes the rest of the network invariant to a
    global phase rotation of the input.

    Args:
        input_channels: number of complex input channels (e.g. ``2`` for the
            LAB / sliding-RGB encodings in the original paper, ``3`` for the
            direct RGB encoding).
        num_classes: number of output classes.
        prototype_size: width of the penultimate feature space (number of
            complex channels reaching the prototype head).
    """

    def __init__(
        self,
        input_channels: int = 2,
        num_classes: int = 10,
        prototype_size: int = 128,
    ) -> None:
        super().__init__()
        self.wfm1 = Conv2d(
            input_channels,
            16,
            kernel_size=3,
            stride=2,
            padding=1,
            padding_mode="reflect",
            bias=False,
        )
        self.diff1 = PhaseDivConv2d(16, kernel_size=3, padding=1)
        self.gtrelu1 = GTReLU(16, phase_scale=True)

        self.wfm2 = Conv2d(
            16,
            32,
            kernel_size=3,
            stride=2,
            padding=1,
            padding_mode="reflect",
            groups=2,
            bias=False,
        )
        self.gtrelu2 = GTReLU(32, phase_scale=True)

        self.wfm3 = Conv2d(
            32,
            64,
            kernel_size=3,
            stride=2,
            padding=1,
            padding_mode="reflect",
            groups=4,
            bias=False,
        )
        self.gtrelu3 = GTReLU(64, phase_scale=True)

        # Global pool via large-kernel conv with full grouping.
        self.wfm4 = Conv2d(64, 64, kernel_size=4, groups=64, bias=False)
        self.fc1 = Conv2d(64, prototype_size, kernel_size=1, groups=4, bias=False)

        # Real BN on the [B, 2*prototype_size] concat-real/imag tensor.
        self.bn = nn.BatchNorm1d(prototype_size * 2)
        self.prototype_size = prototype_size

        self.head = PrototypeDistance(prototype_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_complex():
            x = x.to(torch.cfloat)
        x = self.wfm1(x)
        x = self.diff1(x)
        x = self.gtrelu1(x)
        x = self.wfm2(x)
        x = self.gtrelu2(x)
        x = self.wfm3(x)
        x = self.gtrelu3(x)
        x = self.wfm4(x)
        x = self.fc1(x)
        # x shape [B, P, H', W'] cfloat — H', W' will normally be 1 after the
        # global-pool conv but we keep this generic.
        b = x.shape[0]
        real_flat = _complex_to_real_flat(x).reshape(b, -1)
        real_flat = self.bn(real_flat)
        x = _real_flat_to_complex(
            real_flat.reshape(b, 2 * self.prototype_size, *x.shape[2:])
        )
        # Squeeze spatial to feed the [B, C] prototype head.
        z = x.flatten(2).mean(dim=2)
        return self.head(z)


class CDSEquivariant(nn.Module):
    r"""
    CDS Equivariant (E-Type) Network
    --------------------------------

    Maintains U(1)-equivariance throughout via :class:`ComplexScaling` +
    :class:`EquivariantPhaseReLU` + :class:`MagBatchNorm2d`. The final
    classifier produces invariant logits by pre-rotating prototypes with the
    sum-pooled reference computed from the features themselves
    (see :class:`complextorch.nn.PrototypeDistance` for the mechanism).
    """

    def __init__(
        self,
        input_channels: int = 2,
        num_classes: int = 10,
        prototype_size: int = 128,
    ) -> None:
        super().__init__()
        self.wfm1 = Conv2d(
            input_channels,
            16,
            kernel_size=3,
            stride=2,
            padding=1,
            padding_mode="reflect",
            bias=False,
        )
        self.s1 = ComplexScaling(16)
        self.t1 = EquivariantPhaseReLU(16)

        self.wfm2 = Conv2d(
            16,
            32,
            kernel_size=3,
            stride=2,
            padding=1,
            padding_mode="reflect",
            groups=2,
            bias=False,
        )
        self.s2 = ComplexScaling(32)
        self.t2 = EquivariantPhaseReLU(32)

        self.wfm3 = Conv2d(
            32,
            64,
            kernel_size=3,
            stride=2,
            padding=1,
            padding_mode="reflect",
            groups=4,
            bias=False,
        )
        self.s3 = ComplexScaling(64)
        self.t3 = EquivariantPhaseReLU(64)

        self.wfm4 = Conv2d(64, 64, kernel_size=4, groups=64, bias=False)
        self.fc1 = Conv2d(64, prototype_size, kernel_size=1, groups=4, bias=False)

        self.bn = MagBatchNorm2d(prototype_size)
        self.prototype_size = prototype_size
        self.head = PrototypeDistance(prototype_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_complex():
            x = x.to(torch.cfloat)
        x = self.wfm1(x)
        x = self.s1(x)
        x = self.t1(x)
        x = self.wfm2(x)
        x = self.s2(x)
        x = self.t2(x)
        x = self.wfm3(x)
        x = self.s3(x)
        x = self.t3(x)
        x = self.wfm4(x)
        x = self.fc1(x)
        x = self.bn(x)

        # x shape [B, P, H', W'] cfloat
        # Reference: sum-pool over channels (then over spatial) → [B, 1] cfloat,
        # rescaled to make the magnitude comparable to per-element features.
        ref = x.sum(dim=1, keepdim=False) / math.sqrt(2.0 * self.prototype_size)
        ref = ref.flatten(1).mean(dim=1, keepdim=True)  # [B, 1]
        # Per-channel features for the prototype head.
        z = x.flatten(2).mean(dim=2)  # [B, P]
        return self.head(z, reference=ref)


# ---------------------------------------------------------------------------
# Real-valued backbone for CDSMSTAR (port of cds/model.py:40-106)
# ---------------------------------------------------------------------------


class _SmallCNN(nn.Module):
    r"""Real-valued ResNet-lite used as the SAR classification backbone.

    Ported verbatim from ``cds/model.py:40-106``. Operates on a real-valued
    input of shape ``[B, in_size, H, W]``.
    """

    def __init__(
        self, groups: int = 5, in_size: int = 15, num_classes: int = 10
    ) -> None:
        super().__init__()
        self.relu = nn.ReLU()
        self.conv_1 = nn.Conv2d(in_size, 30, kernel_size=5, groups=groups)
        self.bn_1 = nn.GroupNorm(5, 30)
        self.res1 = nn.Sequential(*self._make_res_block(30, 40))
        self.id1 = nn.Conv2d(30, 40, kernel_size=1)
        self.mp_1 = nn.MaxPool2d(2)
        self.conv_2 = nn.Conv2d(40, 50, kernel_size=5, stride=3, groups=groups)
        self.bn_2 = nn.GroupNorm(10, 50)
        self.res2 = nn.Sequential(*self._make_res_block(50, 60))
        self.id2 = nn.Conv2d(50, 60, kernel_size=1)
        self.conv_3 = nn.Conv2d(60, 70, kernel_size=2, groups=groups)
        self.bn_3 = nn.GroupNorm(14, 70)
        self.linear_2 = nn.Linear(70, 30)
        self.linear_4 = nn.Linear(30, num_classes)

    @staticmethod
    def _make_res_block(in_channel: int, out_channel: int):
        bottleneck = out_channel // 4
        return [
            nn.GroupNorm(5, in_channel),
            nn.ReLU(),
            nn.Conv2d(in_channel, bottleneck, kernel_size=1, bias=False),
            nn.GroupNorm(5, bottleneck),
            nn.ReLU(),
            nn.Conv2d(bottleneck, bottleneck, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(5, bottleneck),
            nn.ReLU(),
            nn.Conv2d(bottleneck, out_channel, kernel_size=1, bias=False),
        ]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_1(x)
        x = self.bn_1(x)
        x_res = self.relu(x)
        x = self.id1(x_res) + self.res1(x_res)
        x = self.mp_1(x)
        x = self.conv_2(x)
        x = self.bn_2(x)
        x_res = self.relu(x)
        x = self.id2(x_res) + self.res2(x_res)
        x = self.conv_3(x)
        x = self.bn_3(x)
        x = self.relu(x)
        x = x.mean(dim=(-1, -2))
        x = self.linear_2(x)
        x = self.relu(x)
        return self.linear_4(x)


class CDSMSTAR(nn.Module):
    r"""
    CDS MSTAR (SAR) Network
    -----------------------

    Complex front-end + real ResNet-lite tail. The complex front-end uses
    :class:`PhaseConjConv2d` for phase-mixing modulation and :class:`GTReLU`
    for nonlinear thresholding; before passing to the real backbone, the
    features are decomposed into ``(log|z|, cos(arg z), sin(arg z))``.

    Args:
        num_classes: number of output classes.
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.wfm1 = Conv2d(1, 5, kernel_size=5, stride=1, bias=False)
        self.diff1 = PhaseConjConv2d(5, kernel_size=3)
        self.gtrelu1 = GTReLU(5)
        self.mp = MagMaxPool2d(2)
        self.wfm2 = Conv2d(5, 5, kernel_size=3, stride=2, bias=False)
        self.gtrelu2 = GTReLU(5)
        # 5 complex channels → 15 real channels (log|z|, cos, sin) per channel
        self.cnn = _SmallCNN(groups=5, in_size=15, num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_complex():
            x = x.to(torch.cfloat)
        x = self.wfm1(x)
        x = self.diff1(x)
        x = self.gtrelu1(x)
        x = self.mp(x)
        x = self.wfm2(x)
        x = self.gtrelu2(x)

        mag = x.abs().clamp(min=1e-5)
        log_mag = torch.log(mag)
        phase = x.angle()
        cos_p = torch.cos(phase)
        sin_p = torch.sin(phase)
        # Stack to [B, 3, C, H, W] then flatten the complex / decomposed dims.
        decomp = torch.stack([log_mag, cos_p, sin_p], dim=1)
        b, three, c, h, w = decomp.shape
        decomp = decomp.reshape(b, three * c, h, w)
        return self.cnn(decomp)
