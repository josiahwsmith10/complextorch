r"""
Pre-Built Complex-Valued Architectures
======================================

Reference architectures composed from the primitives in :mod:`complextorch.nn`.
"""

from complextorch.models.vit import ViT, ViTLayer, vit_t, vit_s, vit_b, vit_l, vit_h
from complextorch.models.cds import CDSInvariant, CDSEquivariant, CDSMSTAR

__all__ = [
    "ViT",
    "ViTLayer",
    "vit_t",
    "vit_s",
    "vit_b",
    "vit_l",
    "vit_h",
    "CDSInvariant",
    "CDSEquivariant",
    "CDSMSTAR",
]
