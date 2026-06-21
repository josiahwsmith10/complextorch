r"""
Pre-Built Complex-Valued Architectures
======================================

Reference architectures composed from the primitives in :mod:`complextorch.nn`.
"""

from complextorch.models.cds import CDSMSTAR, CDSEquivariant, CDSInvariant
from complextorch.models.kan import CVKAN
from complextorch.models.steinmetz import AnalyticNeuralNetwork, SteinmetzNetwork
from complextorch.models.vit import ViT, ViTLayer, vit_b, vit_h, vit_l, vit_s, vit_t

__all__ = [
    "CDSMSTAR",
    "CVKAN",
    "AnalyticNeuralNetwork",
    "CDSEquivariant",
    "CDSInvariant",
    "SteinmetzNetwork",
    "ViT",
    "ViTLayer",
    "vit_b",
    "vit_h",
    "vit_l",
    "vit_s",
    "vit_t",
]
