"""Tests for the CINEView / AccFactor enums in the datasets registry."""

from __future__ import annotations

from complextorch.datasets import AccFactor, CINEView


def test_cineview_members():
    assert CINEView.SAX.value == "SAX"
    assert CINEView.LAX.value == "LAX"
    assert list(CINEView) == [CINEView.SAX, CINEView.LAX]


def test_accfactor_members():
    assert AccFactor.R4.value == 4
    assert AccFactor.R8.value == 8
    assert AccFactor.R10.value == 10


def test_cineview_str_compat():
    assert CINEView.SAX == "SAX"


def test_accfactor_int_compat():
    assert AccFactor.R4 == 4
