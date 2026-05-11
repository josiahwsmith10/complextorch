"""Tests for the SAMPLE in-memory synthetic dataset."""

from __future__ import annotations

import torch

from complextorch.datasets import SAMPLE


def test_sample_basic():
    ds = SAMPLE(num_samples=8, channels=2, height=4, width=4, num_classes=5, seed=42)
    assert len(ds) == 8
    chip, label = ds[0]
    assert chip.shape == (2, 4, 4)
    assert chip.is_complex()
    assert isinstance(label, int)
    assert 0 <= label < 5


def test_sample_reproducibility():
    a = SAMPLE(num_samples=4, height=8, width=8, seed=7)
    b = SAMPLE(num_samples=4, height=8, width=8, seed=7)
    torch.testing.assert_close(a[2][0], b[2][0])


def test_sample_transform_applied():
    sentinel = torch.zeros(1, 4, 4, dtype=torch.cfloat)
    ds = SAMPLE(num_samples=2, height=4, width=4, transform=lambda _x: sentinel)
    chip, _ = ds[1]
    torch.testing.assert_close(chip, sentinel)


def test_sample_root_ignored():
    """root is accepted for API parity but not required to exist."""
    ds = SAMPLE(root="/nonexistent/path", num_samples=2)
    assert len(ds) == 2
