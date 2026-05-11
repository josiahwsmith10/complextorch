r"""
Complex-Valued Dataset Loaders
==============================

A collection of SAR, MRI, and other naturally-complex-valued datasets,
mirroring the dataset surface of :mod:`torchcvnn.datasets`. Install the
optional dependencies with::

    pip install complextorch[datasets]            # h5py for MICCAI MRI
    pip install complextorch[datasets-alos]       # rasterio (GDAL) for ALOS-2

Dataset loaders that require optional dependencies raise a clear
:class:`ImportError` at instantiation time if the dep is not installed.

Most concrete loaders here are adapted from :mod:`torchcvnn.datasets`
(Levi, Dhédin, Fix, Gabot, Durand, Nguyen, Ren — MIT-licensed); per-file
attribution is preserved.
"""

# Each dataset is wrapped in a try/except to keep imports cheap and to
# defer optional-dep errors to dataset instantiation, not import time.
from complextorch.datasets._registry import (
    MICCAI2023,
    S1SLC,
    SAMPLE,
    AccFactor,
    ALOSDataset,
    ATRNetSTAR,
    Bretigny,
    CINEView,
    LeaderFile,
    MSTARTargets,
    PolSFDataset,
    SARImage,
    SLCDataset,
    TrailerFile,
    VolFile,
)

__all__ = [
    "MICCAI2023",
    "S1SLC",
    "SAMPLE",
    "ALOSDataset",
    "ATRNetSTAR",
    "AccFactor",
    "Bretigny",
    "CINEView",
    "LeaderFile",
    "MSTARTargets",
    "PolSFDataset",
    "SARImage",
    "SLCDataset",
    "TrailerFile",
    "VolFile",
]
