"""Tests for the generic SLCDataset reader."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from complextorch.datasets import SLCDataset


def _seed_npy_dir(root, n_files=3, shape=(1, 4, 4)):
    for i in range(n_files):
        arr = (np.random.randn(*shape) + 1j * np.random.randn(*shape)).astype(
            np.complex64
        )
        np.save(root / f"chip_{i:03d}.npy", arr)


def test_slc_dataset_npy(tmp_path):
    _seed_npy_dir(tmp_path, n_files=4)
    ds = SLCDataset(tmp_path)
    assert len(ds) == 4
    chip = ds[0]
    assert chip.shape == (1, 4, 4)
    assert chip.is_complex()


def test_slc_dataset_pt(tmp_path):
    for i in range(3):
        t = torch.randn(1, 4, 4, dtype=torch.cfloat)
        torch.save(t, tmp_path / f"chip_{i:03d}.pt")
    ds = SLCDataset(tmp_path, suffix=".pt")
    assert len(ds) == 3
    chip = ds[1]
    assert chip.is_complex()


def test_slc_dataset_with_annotations(tmp_path):
    _seed_npy_dir(tmp_path, n_files=3)
    ann = tmp_path / "labels.txt"
    ann.write_text("0\n1\n2\n")
    ds = SLCDataset(tmp_path, annotation_file=ann)
    chip, label = ds[2]
    assert label == 2
    assert chip.is_complex()


def test_slc_dataset_with_blank_lines_in_annotations(tmp_path):
    _seed_npy_dir(tmp_path, n_files=2)
    ann = tmp_path / "labels.txt"
    ann.write_text("0\n\n1\n\n")  # blanks filtered
    ds = SLCDataset(tmp_path, annotation_file=ann)
    assert len(ds.labels) == 2


def test_slc_dataset_missing_root_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="root not found"):
        SLCDataset(tmp_path / "does_not_exist")


def test_slc_dataset_annotation_mismatch_raises(tmp_path):
    _seed_npy_dir(tmp_path, n_files=2)
    ann = tmp_path / "labels.txt"
    ann.write_text("0\n1\n2\n")  # 3 labels, 2 files
    with pytest.raises(ValueError, match="annotation count"):
        SLCDataset(tmp_path, annotation_file=ann)


def test_slc_dataset_transform_applied(tmp_path):
    _seed_npy_dir(tmp_path, n_files=2)
    sentinel = torch.ones(1, 4, 4, dtype=torch.cfloat) * 5.0
    ds = SLCDataset(tmp_path, transform=lambda _x: sentinel)
    chip = ds[0]
    torch.testing.assert_close(chip, sentinel)
