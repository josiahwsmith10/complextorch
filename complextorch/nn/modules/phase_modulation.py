r"""
Phase-Modulation Layers (CDS)
=============================

Two related layers that modulate a complex input :math:`x` by a learned
complex function :math:`g(x)` of itself:

- :class:`PhaseDivConv{1,2,3}d` (``y = x / g(x)``) — U(1)-invariant: a global
  phase rotation :math:`e^{j\psi}` of :math:`x` cancels in numerator and
  denominator, so the output is unchanged by global phase. Output magnitude
  is :math:`|x| / |g(x)|`.

- :class:`PhaseConjConv{1,2,3}d` (``y = x \cdot \overline{g(x)}``) — also
  U(1)-invariant when the inner :math:`g` is complex-linear (the cfloat-native
  case here): both factors rotate by :math:`e^{\pm j\psi}` and cancel. Output
  magnitude is :math:`|x| \cdot |g(x)|`. Use this when you want phase
  invariance plus a learned magnitude scaling that grows with :math:`|g(x)|`
  rather than shrinks with :math:`1/|g(x)|`.

Both share an inner complex convolution :math:`g`. When ``use_one_filter=True``
(the default in the reference implementation), :math:`g` has a single output
channel that is broadcast across input channels.

.. note::
    The CDS paper described ``ConjugateLayer`` as a "phase-mixing" operator.
    That characterisation was specific to the paper's two-real-conv
    decomposition of complex convolution; a fully C-linear inner conv (used
    here for compatibility with ``complextorch``'s native cfloat convention)
    yields strict U(1)-invariance instead.

Based on work from the following paper:

    **U. Singhal, Y. Xing, S. X. Yu. Co-Domain Symmetry for Complex-Valued Deep Learning.**

        - CVPR 2022 — ``DivLayer`` and ``ConjugateLayer`` in the reference implementation

        - https://openaccess.thecvf.com/content/CVPR2022/papers/Singhal_Co-Domain_Symmetry_for_Complex-Valued_Deep_Learning_CVPR_2022_paper.pdf
"""

from typing import Tuple

import torch
import torch.nn as nn

from complextorch.nn.modules.conv import Conv1d, Conv2d, Conv3d

__all__ = [
    "PhaseDivConv1d",
    "PhaseDivConv2d",
    "PhaseDivConv3d",
    "PhaseConjConv1d",
    "PhaseConjConv2d",
    "PhaseConjConv3d",
]


def _center_crop(x: torch.Tensor, target_spatial: Tuple[int, ...]) -> torch.Tensor:
    """Center-crop the trailing spatial dims of ``x`` to ``target_spatial``.

    Used when the inner conv ``g`` has a kernel larger than 1 and no padding,
    leaving ``g(x)`` smaller than ``x``. Returns ``x`` unchanged if shapes
    already match.
    """
    spatial_in = x.shape[-len(target_spatial) :]
    if tuple(spatial_in) == tuple(target_spatial):
        return x
    slices = [slice(None), slice(None)]
    for in_size, out_size in zip(spatial_in, target_spatial):
        start = (in_size - out_size) // 2
        slices.append(slice(start, start + out_size))
    return x[tuple(slices)]


class _PhaseModulationNd(nn.Module):
    r"""Shared base for :class:`PhaseDivConv{1,2,3}d` and
    :class:`PhaseConjConv{1,2,3}d`. Subclasses override :meth:`_combine`."""

    _conv_classes = (None, Conv1d, Conv2d, Conv3d)  # indexed by nd

    def __init__(
        self,
        nd: int,
        in_channels: int,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups: int = 1,
        use_one_filter: bool = True,
        eps: float = 1e-7,
    ) -> None:
        super().__init__()
        if nd not in (1, 2, 3):
            raise ValueError(f"nd must be 1, 2, or 3, got {nd}")
        self.nd = nd
        self.in_channels = in_channels
        self.use_one_filter = use_one_filter
        self.eps = eps

        out_channels = 1 if use_one_filter else in_channels
        conv_cls = self._conv_classes[nd]
        self.conv = conv_cls(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=False,
        )

    def _combine(self, x: torch.Tensor, g_x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if not input.is_complex():
            input = input.to(torch.cfloat)
        g_x = self.conv(input)
        if self.use_one_filter:
            # g_x has 1 output channel; expand back to in_channels for elementwise ops.
            g_x = g_x.expand(g_x.shape[0], self.in_channels, *g_x.shape[2:])
        # Center-crop input to g_x's spatial size when the inner conv shrank it.
        target_spatial = g_x.shape[-self.nd :]
        x = _center_crop(input, target_spatial)
        return self._combine(x, g_x)

    def extra_repr(self) -> str:
        return (
            f"in_channels={self.in_channels}, use_one_filter={self.use_one_filter}, "
            f"eps={self.eps}"
        )


class _PhaseDivConvNd(_PhaseModulationNd):
    r"""``y = x · conj(g(x)) / (|g(x)|² + ε)`` — U(1)-invariant."""

    def _combine(self, x: torch.Tensor, g_x: torch.Tensor) -> torch.Tensor:
        denom = (g_x.real * g_x.real + g_x.imag * g_x.imag + self.eps).to(x.dtype)
        return x * g_x.conj() / denom


class _PhaseConjConvNd(_PhaseModulationNd):
    r"""``y = x · conj(g(x))`` — phase-mixing modulator."""

    def _combine(self, x: torch.Tensor, g_x: torch.Tensor) -> torch.Tensor:
        return x * g_x.conj()


def _make_doc(name: str, op: str, invariance: str) -> str:
    return f"""
{name}
{"-" * len(name)}

Phase modulation by a learned complex convolution:

.. math::

    y = {op}

{invariance}

See :mod:`complextorch.nn.modules.phase_modulation` for the shared
construction conventions (``use_one_filter`` default, inner conv reuse, etc.).

Args:
    in_channels: number of complex input channels.
    kernel_size, stride, padding, dilation, groups: forwarded to the inner
        :class:`complextorch.nn.Conv{{1,2,3}}d`.
    use_one_filter: if ``True`` (default), the inner conv produces a single
        complex channel that is broadcast across input channels (this matches
        the CDS reference for the ``I``-type network).
    eps: numerical floor on the denominator (only used by the Div variant).
"""


class PhaseDivConv1d(_PhaseDivConvNd):
    __doc__ = _make_doc(
        "1-D Phase-Division Modulation",
        r"x \cdot \overline{g(x)} / (|g(x)|^2 + \varepsilon)",
        "**U(1)-invariant** under a global phase rotation of ``x``.",
    )

    def __init__(self, in_channels: int, kernel_size, **kwargs) -> None:
        super().__init__(1, in_channels, kernel_size, **kwargs)


class PhaseDivConv2d(_PhaseDivConvNd):
    __doc__ = _make_doc(
        "2-D Phase-Division Modulation",
        r"x \cdot \overline{g(x)} / (|g(x)|^2 + \varepsilon)",
        "**U(1)-invariant** under a global phase rotation of ``x``.",
    )

    def __init__(self, in_channels: int, kernel_size, **kwargs) -> None:
        super().__init__(2, in_channels, kernel_size, **kwargs)


class PhaseDivConv3d(_PhaseDivConvNd):
    __doc__ = _make_doc(
        "3-D Phase-Division Modulation",
        r"x \cdot \overline{g(x)} / (|g(x)|^2 + \varepsilon)",
        "**U(1)-invariant** under a global phase rotation of ``x``.",
    )

    def __init__(self, in_channels: int, kernel_size, **kwargs) -> None:
        super().__init__(3, in_channels, kernel_size, **kwargs)


class PhaseConjConv1d(_PhaseConjConvNd):
    __doc__ = _make_doc(
        "1-D Phase-Conjugate Modulation",
        r"x \cdot \overline{g(x)}",
        "Phase-mixing modulator; magnitude is scaled by :math:`|g(x)|`.",
    )

    def __init__(self, in_channels: int, kernel_size, **kwargs) -> None:
        super().__init__(1, in_channels, kernel_size, **kwargs)


class PhaseConjConv2d(_PhaseConjConvNd):
    __doc__ = _make_doc(
        "2-D Phase-Conjugate Modulation",
        r"x \cdot \overline{g(x)}",
        "Phase-mixing modulator; magnitude is scaled by :math:`|g(x)|`.",
    )

    def __init__(self, in_channels: int, kernel_size, **kwargs) -> None:
        super().__init__(2, in_channels, kernel_size, **kwargs)


class PhaseConjConv3d(_PhaseConjConvNd):
    __doc__ = _make_doc(
        "3-D Phase-Conjugate Modulation",
        r"x \cdot \overline{g(x)}",
        "Phase-mixing modulator; magnitude is scaled by :math:`|g(x)|`.",
    )

    def __init__(self, in_channels: int, kernel_size, **kwargs) -> None:
        super().__init__(3, in_channels, kernel_size, **kwargs)
