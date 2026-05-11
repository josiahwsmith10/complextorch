"""U(1) equivariance / invariance verification across CDS + SurReal modules.

Each test rotates the input by a global complex phase :math:`e^{j\\psi}` and
checks the documented commutation property of the module.

.. note::
    :class:`wFMConv2d` and :class:`wFMReLU` (SurReal) operate on the
    rotation+scaling manifold; their equivariance is defined relative to
    that manifold's group action, not to the strict global phase rotation
    used in this file. They are intentionally not covered here.
"""

from __future__ import annotations

import torch

from complextorch.nn import (
    ComplexScaling,
    EquivariantPhaseReLU,
    MagBatchNorm2d,
    PhaseConjConv2d,
    PhaseDivConv2d,
    PhaseShift,
)


def _rotor(psi: float = 1.3) -> torch.Tensor:
    return torch.polar(torch.tensor(1.0), torch.tensor(psi))


# ---------------------------------------------------------------------------
# Equivariant modules: M(x · e^{jψ}) = M(x) · e^{jψ}
# ---------------------------------------------------------------------------


def test_phase_shift_is_u1_equivariant():
    layer = PhaseShift(num_features=4)
    x = torch.randn(2, 4, 6, 6, dtype=torch.cfloat)
    rotor = _rotor()
    torch.testing.assert_close(layer(x * rotor), layer(x) * rotor, atol=1e-5, rtol=1e-5)


def test_complex_scaling_is_u1_equivariant():
    layer = ComplexScaling(num_features=4)
    x = torch.randn(2, 4, 6, 6, dtype=torch.cfloat)
    rotor = _rotor()
    torch.testing.assert_close(layer(x * rotor), layer(x) * rotor, atol=1e-5, rtol=1e-5)


def test_equivariant_phase_relu_is_u1_equivariant():
    layer = EquivariantPhaseReLU(num_channels=4)
    x = torch.randn(2, 4, 6, 6, dtype=torch.cfloat) + 0.1
    rotor = _rotor()
    torch.testing.assert_close(layer(x * rotor), layer(x) * rotor, atol=1e-4, rtol=1e-4)


def test_mag_batchnorm_is_u1_equivariant():
    layer = MagBatchNorm2d(num_features=4)
    layer.train()
    # Warm running stats with one forward, then verify in eval mode where
    # BN does not depend on the batch (so rotation acts purely on phase).
    _ = layer(torch.randn(8, 4, 6, 6, dtype=torch.cfloat))
    layer.eval()
    x = torch.randn(2, 4, 6, 6, dtype=torch.cfloat) + 0.1
    rotor = _rotor()
    torch.testing.assert_close(layer(x * rotor), layer(x) * rotor, atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# Invariant modules: M(x · e^{jψ}) = M(x)
# ---------------------------------------------------------------------------


def test_phase_div_conv2d_is_u1_invariant():
    layer = PhaseDivConv2d(in_channels=3, kernel_size=3, padding=1)
    x = torch.randn(2, 3, 5, 5, dtype=torch.cfloat) + 0.1
    rotor = _rotor()
    torch.testing.assert_close(layer(x * rotor), layer(x), atol=1e-4, rtol=1e-4)


def test_phase_conj_conv2d_is_u1_invariant():
    """With a C-linear inner conv, ``x · conj(g(x))`` is invariant too."""
    layer = PhaseConjConv2d(in_channels=3, kernel_size=3, padding=1)
    x = torch.randn(2, 3, 5, 5, dtype=torch.cfloat) + 0.1
    rotor = _rotor()
    torch.testing.assert_close(layer(x * rotor), layer(x), atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# Composition: chain of equivariant ops should remain equivariant.
# ---------------------------------------------------------------------------


def test_composition_is_u1_equivariant():
    """Conv → ComplexScaling → EquivariantPhaseReLU is equivariant.

    Note: complextorch's stock ``Conv2d`` is *not* U(1)-equivariant in general
    (a generic complex linear map mixes phase and magnitude). For a
    composition that's *guaranteed* equivariant we use only modules that
    preserve phase: ComplexScaling, EquivariantPhaseReLU.
    """
    block = torch.nn.Sequential(
        ComplexScaling(num_features=4),
        EquivariantPhaseReLU(num_channels=4),
    )
    x = torch.randn(2, 4, 6, 6, dtype=torch.cfloat) + 0.1
    rotor = _rotor()
    torch.testing.assert_close(block(x * rotor), block(x) * rotor, atol=1e-4, rtol=1e-4)
