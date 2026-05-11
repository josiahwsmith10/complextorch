"""Tests for split Type-B (polar) activation functions."""

from __future__ import annotations


import pytest
import torch
import torch.nn as nn

from complextorch.nn.modules.activation.split_type_B import (
    AdaptiveModReLU,
    CVPolarLog,
    CVPolarSquash,
    CVPolarTanh,
    GeneralizedPolarActivation,
    modReLU,
)


@pytest.fixture
def z_channel():
    # (B, C, L) for AdaptiveModReLU
    return torch.randn(2, 3, 6, dtype=torch.cfloat)


@pytest.mark.parametrize("cls", [CVPolarTanh, CVPolarSquash, CVPolarLog])
def test_polar_activations_preserve_phase(cls):
    z = torch.randn(8, dtype=torch.cfloat) + 1e-3  # avoid |z|=0
    out = cls()(z)
    # Phase should be unchanged (these all use phase_fun=None)
    torch.testing.assert_close(out.angle(), z.angle(), atol=1e-5, rtol=1e-5)


def test_cvpolar_squash_magnitude_bounded():
    z = torch.randn(16, dtype=torch.cfloat)
    out = CVPolarSquash()(z)
    # squash: x^2 / (1 + x^2) -> magnitude < 1
    assert out.abs().max().item() < 1.0


def test_cvpolar_log_monotonic_in_magnitude():
    z = torch.tensor([1.0 + 0j, 2.0 + 0j, 5.0 + 0j])
    out = CVPolarLog()(z)
    # log(|z|+1) is monotonic increasing
    mags = out.abs()
    assert mags[0] < mags[1] < mags[2]


def test_modrelu_static_bias_must_be_negative():
    with pytest.raises(AssertionError, match="smaller than 0"):
        modReLU(bias=0.5)


def test_modrelu_static_bias_zeros_small_magnitudes():
    act = modReLU(bias=-0.5)
    z = torch.tensor([0.1 + 0j, 1.0 + 0j], dtype=torch.cfloat)
    out = act(z)
    assert out[0].abs().item() == 0.0  # |0.1| - 0.5 < 0 -> zeroed
    assert out[1].abs().item() > 0.0


def test_modrelu_learnable_bias_can_be_positive():
    act = modReLU(bias=0.1, learnable=True)
    assert isinstance(act.activation_mag.bias, nn.Parameter)


def test_modrelu_static_bias_is_buffer():
    act = modReLU(bias=-0.1, learnable=False)
    assert not isinstance(act.activation_mag.bias, nn.Parameter)
    assert "bias" in dict(act.activation_mag.named_buffers())


def test_adaptive_modrelu_per_channel(z_channel):
    act = AdaptiveModReLU(num_features=3, init=-0.1)
    out = act(z_channel)
    assert out.shape == z_channel.shape
    assert out.is_complex()
    assert act.activation_mag.bias.shape == (3,)


def test_generalized_polar_with_phase_fun():
    z = torch.randn(4, dtype=torch.cfloat) + 1e-3
    # Use a non-trivial phase function
    act = GeneralizedPolarActivation(nn.Identity(), nn.Identity())
    out = act(z)
    # phase passed through identity -> phase same; magnitude unchanged
    torch.testing.assert_close(out.abs(), z.abs(), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(out.angle(), z.angle(), atol=1e-5, rtol=1e-5)
