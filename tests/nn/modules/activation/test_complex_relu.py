"""Tests for complex ReLU variants."""

from __future__ import annotations

import math

import pytest
import torch

from complextorch.nn.modules.activation.complex_relu import (
    CPReLU,
    CReLU,
    CVSplitReLU,
    EquivariantPhaseReLU,
    GTReLU,
    _PhaseHalfPlaneMask,
    zAbsReLU,
    zLeakyReLU,
)


@pytest.fixture
def z():
    return torch.randn(4, 8, dtype=torch.cfloat)


@pytest.mark.parametrize("cls", [CVSplitReLU, CReLU])
def test_split_relu_zeros_negatives(cls, z):
    out = cls(inplace=False)(z.clone())
    assert (out.real >= 0).all()
    assert (out.imag >= 0).all()


@pytest.mark.parametrize("cls", [CVSplitReLU, CReLU])
def test_split_relu_default_is_out_of_place(cls, z):
    """Default inplace=False: input unmodified, leaf tensors with grad work."""
    original = z.clone()
    cls()(z)
    torch.testing.assert_close(z, original)
    leaf = torch.randn(4, dtype=torch.cfloat, requires_grad=True)
    out = cls()(leaf)  # must not raise on a leaf requiring grad
    out.abs().sum().backward()
    assert torch.isfinite(leaf.grad).all()


def test_split_relu_inplace_opt_in(z):
    """inplace=True remains available and rectifies the input itself."""
    out = CVSplitReLU(inplace=True)(z)
    assert (z.real >= 0).all()
    assert (z.imag >= 0).all()
    torch.testing.assert_close(out, z)


def test_cprelu_learnable(z):
    act = CPReLU()
    out = act(z)
    assert out.shape == z.shape
    # Two PReLU modules, each with a learnable weight
    params = list(act.parameters())
    assert len(params) == 2


def test_zabsrelu_threshold_zero_passes_all():
    z = torch.randn(8, dtype=torch.cfloat)
    out = zAbsReLU(a_init=0.0)(z)
    torch.testing.assert_close(out, z)


def test_zabsrelu_high_threshold_zeros_small():
    z = torch.tensor([0.1 + 0j, 5.0 + 0j], dtype=torch.cfloat)
    out = zAbsReLU(a_init=1.0)(z)
    assert out[0].abs().item() == 0.0
    assert out[1].abs().item() == 5.0


def test_zabsrelu_real_input_path():
    """Real (non-complex) input works through the same gate."""
    x = torch.tensor([0.1, 5.0])  # real, not complex
    out = zAbsReLU(a_init=1.0)(x)
    assert out[0].abs().item() == 0.0
    assert out[1].item() == 5.0
    assert not out.is_complex()


def test_zabsrelu_forward_is_bit_exact_hard_gate():
    """The STE must not perturb the forward pass: bit-exact hard gate."""
    z = torch.randn(64, dtype=torch.cfloat)
    layer = zAbsReLU(a_init=1.0)
    expected = z * (z.abs() >= 1.0).to(torch.float32)
    assert torch.equal(layer(z), expected)


def test_zabsrelu_no_grad_matches_grad_path():
    """Under no_grad the surrogate is skipped, but the output is unchanged."""
    z = torch.randn(64, dtype=torch.cfloat)
    layer = zAbsReLU(a_init=0.5)
    grad_out = layer(z)  # grad-enabled: goes through the STE surrogate
    with torch.no_grad():
        no_grad_out = layer(z)  # skips the surrogate entirely
    assert torch.equal(grad_out.detach(), no_grad_out)


def test_zabsrelu_threshold_receives_gradient():
    """The learnable threshold gets a finite, nonzero surrogate gradient."""
    z = torch.tensor([0.5 + 0j, 0.9 + 0.1j, 1.1 + 0j, 2.0 - 1.0j])
    layer = zAbsReLU(a_init=1.0)
    layer(z).abs().sum().backward()
    assert layer.a.grad is not None
    assert torch.isfinite(layer.a.grad).all()
    assert layer.a.grad.item() != 0.0


def test_zabsrelu_gradient_at_default_init():
    """Even at the identity init a_init=0.0, `a` can still learn."""
    z = torch.tensor([0.05 + 0j, 0.1 + 0.1j, 0.2 + 0j])
    layer = zAbsReLU()
    layer(z).abs().sum().backward()
    assert torch.isfinite(layer.a.grad).all()
    assert layer.a.grad.item() != 0.0


def test_zabsrelu_input_grad_finite_at_zero():
    """torch's sgn-based abs backward keeps an exact-zero element finite."""
    z = torch.tensor([0.0 + 0j, 1.0 + 1.0j], requires_grad=True)
    out = zAbsReLU(a_init=0.5)(z)
    out.abs().sum().backward()
    assert torch.isfinite(z.grad.real).all()
    assert torch.isfinite(z.grad.imag).all()


def test_zabsrelu_extra_repr_and_state_dict():
    layer = zAbsReLU(a_init=0.3, tau=0.05)
    assert "tau=0.05" in layer.extra_repr()
    assert "a" in layer.state_dict()


def test_zleakyrelu_first_quadrant_passes():
    z = torch.tensor([1 + 1j, -1 + 1j, 1 - 1j, -1 - 1j], dtype=torch.cfloat)
    out = zLeakyReLU(negative_slope=0.5)(z)
    torch.testing.assert_close(out[0], z[0])
    torch.testing.assert_close(out[1], 0.5 * z[1])
    torch.testing.assert_close(out[2], 0.5 * z[2])
    torch.testing.assert_close(out[3], 0.5 * z[3])


def test_zleakyrelu_extra_repr():
    s = zLeakyReLU(negative_slope=0.2).extra_repr()
    assert "0.2" in s


# ---------- _PhaseHalfPlaneMask (custom autograd) ----------


def test_phase_halfplane_mask_zeros_lower_halfplane():
    phase = torch.tensor([0.0, math.pi / 2, math.pi, 3 * math.pi / 2, -math.pi / 4])
    out = _PhaseHalfPlaneMask.apply(phase)
    # 0, pi/2, pi → in [0, pi] (kept). 3pi/2 → mod 2pi = 3pi/2 (not kept).
    # -pi/4 → mod 2pi = 7pi/4 (not kept).
    assert out[0].item() == pytest.approx(0.0)
    assert out[1].item() == pytest.approx(math.pi / 2)
    assert out[2].item() == pytest.approx(math.pi)
    assert out[3].item() == pytest.approx(0.0)
    assert out[4].item() == pytest.approx(0.0)


def test_phase_halfplane_mask_gradient_is_mask():
    phase = torch.tensor([math.pi / 4, 3 * math.pi / 2], requires_grad=True)
    out = _PhaseHalfPlaneMask.apply(phase)
    out.sum().backward()
    # Gradient is 1 where the mask is 1, 0 elsewhere
    assert phase.grad[0].item() == pytest.approx(1.0)
    assert phase.grad[1].item() == pytest.approx(0.0)


# ---------- GTReLU ----------


def test_gtrelu_forward_shape():
    layer = GTReLU(num_channels=4)
    x = torch.randn(2, 4, 6, 6, dtype=torch.cfloat)
    out = layer(x)
    assert out.shape == x.shape
    assert out.is_complex()


def test_gtrelu_phase_scale_adds_parameter():
    layer = GTReLU(num_channels=4, phase_scale=True)
    assert layer.lambd is not None
    assert layer.lambd.shape == (4,)


def test_gtrelu_no_phase_scale_no_lambda():
    layer = GTReLU(num_channels=4, phase_scale=False)
    assert layer.lambd is None


def test_gtrelu_global_scaling():
    layer = GTReLU(num_channels=4, global_scaling=True)
    assert layer.alpha.shape == (1,)
    assert layer.beta.shape == (1,)


def test_gtrelu_grad_flows():
    layer = GTReLU(num_channels=3, phase_scale=True)
    x = torch.randn(2, 3, 4, dtype=torch.cfloat, requires_grad=True)
    out = layer(x)
    out.abs().sum().backward()
    assert torch.isfinite(layer.alpha.grad).all()
    assert torch.isfinite(layer.lambd.grad).all()


# ---------- EquivariantPhaseReLU ----------


def test_equivariant_phase_relu_forward_shape():
    layer = EquivariantPhaseReLU(num_channels=4)
    x = torch.randn(2, 4, 6, 6, dtype=torch.cfloat)
    out = layer(x)
    assert out.shape == x.shape
    assert out.is_complex()


def test_equivariant_phase_relu_is_u1_equivariant():
    """Rotating the input by a global phase rotates the output by the same."""
    layer = EquivariantPhaseReLU(num_channels=4)
    x = torch.randn(2, 4, 5, 5, dtype=torch.cfloat) + 0.1
    rotor = torch.polar(torch.tensor(1.0), torch.tensor(1.3))
    y1 = layer(x * rotor)
    y2 = layer(x) * rotor
    torch.testing.assert_close(y1, y2, atol=1e-4, rtol=1e-4)


def test_equivariant_phase_relu_grad_flows():
    layer = EquivariantPhaseReLU(num_channels=3)
    x = torch.randn(2, 3, 4, dtype=torch.cfloat, requires_grad=True)
    out = layer(x)
    out.abs().sum().backward()
    assert torch.isfinite(layer.phase_gain.grad).all()


def test_broadcast_channelwise_passthrough_for_1d_input():
    """1-D input -> the else branch returns t unchanged."""
    from complextorch.nn.modules.activation.complex_relu import _broadcast_channelwise

    t = torch.zeros(4)
    out = _broadcast_channelwise(t, input_dim=1)
    assert out is t


def test_gtrelu_real_input_auto_casts():
    from complextorch.nn.modules.activation.complex_relu import GTReLU

    layer = GTReLU(num_channels=4)
    x = torch.randn(2, 4, 6)
    out = layer(x)
    assert out.is_complex()


def test_gtrelu_extra_repr():
    from complextorch.nn.modules.activation.complex_relu import GTReLU

    s = GTReLU(num_channels=4, global_scaling=True, phase_scale=True).extra_repr()
    assert "num_channels=4" in s
    assert "phase_scale=True" in s


def test_equivariant_phase_relu_real_input_auto_casts():
    layer = EquivariantPhaseReLU(num_channels=3)
    x = torch.randn(2, 3, 4)
    out = layer(x)
    assert out.is_complex()


def test_equivariant_phase_relu_extra_repr():
    s = EquivariantPhaseReLU(num_channels=4).extra_repr()
    assert "num_channels=4" in s
