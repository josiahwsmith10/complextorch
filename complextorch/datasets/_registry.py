r"""
Dataset Class Registry
======================

Stub / minimal-port versions of the dataset loaders. Most of these wrap
file-format readers that are non-trivial to maintain in lockstep with the
upstream sibling library :mod:`torchcvnn.datasets`. The classes below
expose the documented constructor signatures and basic ``__len__`` /
``__getitem__`` behavior; for the heavier SAR/MRI formats, instantiation
raises :class:`NotImplementedError` with a clear pointer to the upstream
reference.

This shape lets users:

- ``from complextorch.datasets import PolSFDataset`` always works (no import-time errors),
- Tab-completion and type checking see the full dataset surface,
- The first time a heavy SAR/MRI loader is instantiated, the user gets a
  precise message pointing to the upstream code or asking for the optional
  dependency.

If you need a fully-ported loader, contributions are welcome — most of the
work is mechanical I/O against the well-documented SAR/MRI file formats.
"""

import enum
import os
from pathlib import Path
from typing import Any, Callable, Optional, Sequence, Tuple, Union

import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _NotImplementedDataset(Dataset):
    r"""Base for datasets whose file-format readers are not yet ported.

    Subclasses set ``_reference`` to the upstream class name for a clear error.
    """

    _reference: str = "torchcvnn.datasets.<XXX>"

    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError(
            f"{type(self).__name__} is not yet implemented in complextorch. "
            f"See the upstream reference implementation at {self._reference!r}; "
            "porting is welcome — see complextorch/datasets/_registry.py."
        )


# ---------------------------------------------------------------------------
# SAMPLE — minimal sanity-test dataset (in-memory random complex tensors)
# ---------------------------------------------------------------------------


class SAMPLE(Dataset):
    r"""
    Minimal in-memory sample dataset of complex tensors.

    Intended as a 'hello-world' dataset for testing pipelines. Generates
    ``num_samples`` random complex chips of shape ``(channels, height,
    width)`` with deterministic seeds, paired with integer class labels.

    Args:
        root: ignored; present for API parity with file-backed datasets.
        num_samples: number of (chip, label) pairs.
        channels: complex channels per chip.
        height: chip height.
        width: chip width.
        num_classes: range of label values.
        transform: optional callable applied to each chip.
        seed: RNG seed for reproducibility.
    """

    def __init__(
        self,
        root: Optional[Union[str, os.PathLike]] = None,
        num_samples: int = 128,
        channels: int = 1,
        height: int = 32,
        width: int = 32,
        num_classes: int = 10,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.num_samples = num_samples
        self.channels = channels
        self.height = height
        self.width = width
        self.num_classes = num_classes
        self.transform = transform
        gen = torch.Generator().manual_seed(seed)
        self._data = (
            torch.randn(num_samples, channels, height, width, generator=gen)
            + 1j * torch.randn(num_samples, channels, height, width, generator=gen)
        ).to(torch.cfloat)
        self._labels = torch.randint(
            0, num_classes, (num_samples,), generator=gen
        ).tolist()

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        chip = self._data[index]
        if self.transform is not None:
            chip = self.transform(chip)
        return chip, self._labels[index]


# ---------------------------------------------------------------------------
# SLCDataset — generic Single-Look Complex file reader
# ---------------------------------------------------------------------------


class SLCDataset(Dataset):
    r"""
    Generic Single-Look Complex (SLC) dataset.

    Loads a directory of ``.npy`` or ``.pt`` files (each a complex tensor of
    shape ``(channels, H, W)``) paired with optional integer labels from a
    text file. The format is intentionally simple to let users plug in
    their own SLC products without writing a custom :class:`Dataset`.

    Args:
        root: directory containing complex tensor files.
        annotation_file: optional path to a text file with one ``label`` per
            line (in the same order as ``sorted(os.listdir(root))``).
        suffix: extension of the tensor files (default ``.npy``).
        transform: optional callable applied to each loaded tensor.
    """

    def __init__(
        self,
        root: Union[str, os.PathLike],
        annotation_file: Optional[Union[str, os.PathLike]] = None,
        suffix: str = ".npy",
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(f"SLCDataset root not found: {self.root}")
        self.files = sorted(p for p in self.root.iterdir() if p.suffix == suffix)
        self.suffix = suffix
        self.transform = transform

        self.labels: Optional[Sequence[int]] = None
        if annotation_file is not None:
            with open(annotation_file, "r") as fh:
                self.labels = [int(line.strip()) for line in fh if line.strip()]
            if len(self.labels) != len(self.files):
                raise ValueError(
                    f"annotation count ({len(self.labels)}) != file count ({len(self.files)})"
                )

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int):
        path = self.files[index]
        if self.suffix == ".npy":
            import numpy as np

            arr = np.load(path)
            chip = torch.as_tensor(arr).to(torch.cfloat)
        else:
            chip = torch.load(path).to(torch.cfloat)
        if self.transform is not None:
            chip = self.transform(chip)
        if self.labels is None:
            return chip
        return chip, self.labels[index]


# ---------------------------------------------------------------------------
# Heavy SAR / MRI loaders — stubs with upstream-attribution errors
# ---------------------------------------------------------------------------


class PolSFDataset(_NotImplementedDataset):
    """San Francisco PolSAR dataset (IETR-Lab). Quad-pol patches."""

    _reference = "torchcvnn.datasets.PolSFDataset"


class Bretigny(_NotImplementedDataset):
    """Bretigny airfield SAR (full polarimetry)."""

    _reference = "torchcvnn.datasets.Bretigny"


class S1SLC(_NotImplementedDataset):
    """Sentinel-1 Single-Look Complex (S1SLCCVDL)."""

    _reference = "torchcvnn.datasets.S1SLC"


class MSTARTargets(_NotImplementedDataset):
    """MSTAR (Moving and Stationary Target Recognition) SAR ATR dataset."""

    _reference = "torchcvnn.datasets.MSTARTargets"


class ATRNetSTAR(_NotImplementedDataset):
    """ATRNet-STAR target-recognition SAR dataset."""

    _reference = "torchcvnn.datasets.ATRNetSTAR"


# ---------------------------------------------------------------------------
# MICCAI 2023 — requires h5py (under [datasets] extra)
# ---------------------------------------------------------------------------


class CINEView(str, enum.Enum):
    SAX = "SAX"
    LAX = "LAX"


class AccFactor(int, enum.Enum):
    R4 = 4
    R8 = 8
    R10 = 10


class MICCAI2023(_NotImplementedDataset):
    """MICCAI 2023 cardiac cine MRI (k-space, complex).

    Requires ``h5py`` (install with ``pip install complextorch[datasets]``).
    """

    _reference = "torchcvnn.datasets.MICCAI2023"


# ---------------------------------------------------------------------------
# ALOS-2 / CEOS — requires rasterio (under [datasets-alos] extra)
# ---------------------------------------------------------------------------


class ALOSDataset(_NotImplementedDataset):
    """ALOS-2 PALSAR dataset.

    Requires ``rasterio`` (install with ``pip install complextorch[datasets-alos]``).
    """

    _reference = "torchcvnn.datasets.ALOSDataset"


class VolFile(_NotImplementedDataset):
    """CEOS Volume Directory File parser."""

    _reference = "torchcvnn.datasets.alos2.VolFile"


class LeaderFile(_NotImplementedDataset):
    """CEOS SAR Leader File parser."""

    _reference = "torchcvnn.datasets.alos2.LeaderFile"


class TrailerFile(_NotImplementedDataset):
    """CEOS SAR Trailer File parser."""

    _reference = "torchcvnn.datasets.alos2.TrailerFile"


class SARImage(_NotImplementedDataset):
    """CEOS SAR Image data handler."""

    _reference = "torchcvnn.datasets.alos2.SARImage"
