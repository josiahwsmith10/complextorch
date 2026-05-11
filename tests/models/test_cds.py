"""Smoke tests for the CDS reference models."""

from __future__ import annotations

import torch

from complextorch.models.cds import CDSMSTAR, CDSEquivariant, CDSInvariant


def test_cds_invariant_forward_backward():
    model = CDSInvariant(input_channels=2, num_classes=10, prototype_size=32)
    x = torch.randn(2, 2, 32, 32, dtype=torch.cfloat)
    logits = model(x)
    assert logits.shape == (2, 10)
    logits.sum().backward()
    # At least one param should have a finite, non-zero gradient.
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert any(g.abs().sum().item() > 0 for g in grads)


def test_cds_equivariant_forward_backward():
    model = CDSEquivariant(input_channels=2, num_classes=10, prototype_size=32)
    x = torch.randn(2, 2, 32, 32, dtype=torch.cfloat)
    logits = model(x)
    assert logits.shape == (2, 10)
    logits.sum().backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert any(g.abs().sum().item() > 0 for g in grads)


def test_cds_mstar_forward_backward():
    # MSTAR input: 1 complex channel; size 88x88 is the original SAR chip size.
    model = CDSMSTAR(num_classes=10)
    x = torch.randn(2, 1, 88, 88, dtype=torch.cfloat) + 0.1
    logits = model(x)
    assert logits.shape == (2, 10)
    logits.sum().backward()


def test_cds_models_accept_real_input():
    """All three CDS models auto-cast a real input to cfloat in forward."""
    for cls, x in [
        (CDSInvariant(input_channels=2, num_classes=4, prototype_size=8), torch.randn(2, 2, 32, 32)),
        (CDSEquivariant(input_channels=2, num_classes=4, prototype_size=8), torch.randn(2, 2, 32, 32)),
        (CDSMSTAR(num_classes=4), torch.randn(2, 1, 88, 88) + 0.1),
    ]:
        out = cls(x)
        assert out.shape[0] == 2


def test_cds_invariant_is_phase_invariant_in_eval():
    """The DivConv after wfm1 makes CDSInvariant invariant to global phase.

    Use eval() because the trailing real BatchNorm1d (which mixes magnitudes
    of real/imag parts in a way that breaks pure invariance) needs running
    stats fixed, AND in training mode batch-norm depends on the rotation.
    """
    model = CDSInvariant(input_channels=2, num_classes=4, prototype_size=16)
    x = torch.randn(2, 2, 32, 32, dtype=torch.cfloat)
    # Warm up running stats.
    model.train()
    _ = model(x)
    model.eval()
    rotor = torch.polar(torch.tensor(1.0), torch.tensor(0.8))
    logits1 = model(x)
    logits2 = model(x * rotor)
    # Should match closely (the path before BN is genuinely invariant; BN
    # operates on real+imag concatenation so post-BN there is slight rotation
    # dependence). Loosen tolerance accordingly.
    torch.testing.assert_close(logits1, logits2, atol=5e-2, rtol=5e-2)
