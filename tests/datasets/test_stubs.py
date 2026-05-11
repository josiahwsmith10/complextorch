"""Tests asserting NotImplementedError stubs for the heavy SAR/MRI loaders."""

from __future__ import annotations

import pytest

from complextorch.datasets import (
    ALOSDataset,
    ATRNetSTAR,
    Bretigny,
    LeaderFile,
    MICCAI2023,
    MSTARTargets,
    PolSFDataset,
    S1SLC,
    SARImage,
    TrailerFile,
    VolFile,
)


@pytest.mark.parametrize(
    "cls",
    [
        PolSFDataset,
        Bretigny,
        S1SLC,
        MSTARTargets,
        ATRNetSTAR,
        MICCAI2023,
        ALOSDataset,
        VolFile,
        LeaderFile,
        TrailerFile,
        SARImage,
    ],
)
def test_stub_raises_not_implemented(cls):
    with pytest.raises(NotImplementedError, match="not yet implemented"):
        cls()


def test_stub_reference_points_upstream():
    """The error message must reference the upstream torchcvnn class."""
    with pytest.raises(NotImplementedError, match="torchcvnn"):
        PolSFDataset()
