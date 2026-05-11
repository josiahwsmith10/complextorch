"""Tests for transforms.functional helpers (polsar_dict_to_array, rescale_intensity)."""

from __future__ import annotations

import pytest
import torch

from complextorch.transforms.functional import polsar_dict_to_array, rescale_intensity


def test_polsar_dict_to_array_default_order():
    d = {
        "HH": torch.randn(4, 4, dtype=torch.cfloat),
        "HV": torch.randn(4, 4, dtype=torch.cfloat),
        "VH": torch.randn(4, 4, dtype=torch.cfloat),
        "VV": torch.randn(4, 4, dtype=torch.cfloat),
    }
    arr = polsar_dict_to_array(d)
    assert arr.shape == (4, 4, 4)
    torch.testing.assert_close(arr[0], d["HH"])


def test_polsar_dict_to_array_partial_order():
    d = {
        "HH": torch.randn(4, 4, dtype=torch.cfloat),
        "VV": torch.randn(4, 4, dtype=torch.cfloat),
    }
    arr = polsar_dict_to_array(d, order=("HH", "VV"))
    assert arr.shape == (2, 4, 4)


def test_polsar_dict_to_array_no_match():
    with pytest.raises(ValueError, match="none of the requested"):
        polsar_dict_to_array({"foo": torch.zeros(4, 4)}, order=("HH",))


def test_rescale_intensity_default_range():
    x = torch.tensor([0.0, 5.0, 10.0])
    out = rescale_intensity(x)
    torch.testing.assert_close(out, torch.tensor([0.0, 0.5, 1.0]))


def test_rescale_intensity_explicit_range():
    x = torch.tensor([2.0, 4.0, 6.0])
    out = rescale_intensity(x, in_range=(0.0, 10.0), out_range=(-1.0, 1.0))
    expected = torch.tensor([-0.6, -0.2, 0.2])
    torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)


def test_rescale_intensity_complex_raises():
    with pytest.raises(TypeError, match="real tensor"):
        rescale_intensity(torch.zeros(3, dtype=torch.cfloat))


def test_rescale_intensity_degenerate_range():
    x = torch.ones(4)
    out = rescale_intensity(x)  # in_range = (1, 1)
    # All values clamp to out_lo (0.0 by default)
    torch.testing.assert_close(out, torch.zeros(4))


def test_rescale_intensity_clamps_values():
    x = torch.tensor([-1.0, 5.0, 11.0])
    out = rescale_intensity(x, in_range=(0.0, 10.0))
    # -1 -> clamp to 0 -> 0.0; 11 -> clamp to 10 -> 1.0
    torch.testing.assert_close(out, torch.tensor([0.0, 0.5, 1.0]))
