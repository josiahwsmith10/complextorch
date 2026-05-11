"""Tests for PrototypeDistance."""

from __future__ import annotations

import pytest
import torch

from complextorch.nn.modules.prototype import PrototypeDistance


def test_prototype_distance_forward_shape():
    head = PrototypeDistance(in_features=8, num_prototypes=10)
    x = torch.randn(4, 8, dtype=torch.cfloat)
    logits = head(x)
    assert logits.shape == (4, 10)
    assert not logits.is_complex()


def test_prototype_distance_closest_prototype_has_largest_logit():
    """If an input matches prototype k exactly, logit k should be largest."""
    head = PrototypeDistance(in_features=4, num_prototypes=3)
    # Match prototype 1 exactly.
    target_proto = head.prototypes[:, 1].detach().clone()
    x = target_proto.unsqueeze(0)  # [1, 4]
    logits = head(x)
    assert logits.argmax(dim=1).item() == 1


def test_prototype_distance_rejects_non_2d_input():
    head = PrototypeDistance(in_features=4, num_prototypes=3)
    with pytest.raises(ValueError, match="expects input of shape"):
        head(torch.randn(2, 4, 6, dtype=torch.cfloat))


def test_prototype_distance_with_reference_scalar_broadcasts():
    """E-type call: a single complex scalar per batch broadcasts over channels."""
    head = PrototypeDistance(in_features=4, num_prototypes=3)
    x = torch.randn(2, 4, dtype=torch.cfloat)
    ref = torch.randn(2, 1, dtype=torch.cfloat)
    logits = head(x, reference=ref)
    assert logits.shape == (2, 3)


def test_prototype_distance_with_reference_per_channel():
    head = PrototypeDistance(in_features=4, num_prototypes=3)
    x = torch.randn(2, 4, dtype=torch.cfloat)
    ref = torch.randn(2, 4, dtype=torch.cfloat)
    logits = head(x, reference=ref)
    assert logits.shape == (2, 3)


def test_prototype_distance_etype_invariant_under_phase_rotation():
    """When both input and reference rotate by e^{j psi}, logits are unchanged."""
    head = PrototypeDistance(in_features=4, num_prototypes=3)
    x = torch.randn(2, 4, dtype=torch.cfloat)
    ref = torch.randn(2, 1, dtype=torch.cfloat)
    rotor = torch.polar(torch.tensor(1.0), torch.tensor(1.1))
    logits1 = head(x, reference=ref)
    logits2 = head(x * rotor, reference=ref * rotor)
    torch.testing.assert_close(logits1, logits2, atol=1e-5, rtol=1e-5)


def test_prototype_distance_grad_flows():
    head = PrototypeDistance(in_features=4, num_prototypes=3)
    x = torch.randn(2, 4, dtype=torch.cfloat, requires_grad=True)
    logits = head(x)
    logits.sum().backward()
    assert head.prototypes.grad is not None
    assert head.temperature.grad is not None
    assert torch.isfinite(head.prototypes.grad).all()


def test_prototype_distance_extra_repr():
    s = PrototypeDistance(in_features=4, num_prototypes=10).extra_repr()
    assert "in_features=4" in s
    assert "num_prototypes=10" in s
