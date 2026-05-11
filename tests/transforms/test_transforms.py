"""Tests for the class-based transforms."""

from __future__ import annotations


import pytest
import torch

from complextorch.transforms import (
    Amplitude,
    CenterCrop,
    FFT2,
    FFTResize,
    HWC2CHW,
    IFFT2,
    LogAmplitude,
    Normalize,
    PadIfNeeded,
    PolSAR,
    RandomPhase,
    RealImaginary,
    SpatialResize,
    ToImaginary,
    ToReal,
    ToTensor,
    Unsqueeze,
)


# ---------- Casting / shape ----------


def test_to_tensor_from_list():
    out = ToTensor()([[1.0, 2.0], [3.0, 4.0]])
    assert out.is_complex()
    assert out.shape == (2, 2)


def test_to_tensor_extra_repr():
    # torch.cfloat is an alias for torch.complex64
    assert "complex64" in ToTensor(dtype=torch.cfloat).extra_repr()


def test_unsqueeze():
    out = Unsqueeze(dim=0)(torch.zeros(3))
    assert out.shape == (1, 3)
    assert "dim=0" in Unsqueeze(0).extra_repr()


def test_hwc2chw():
    x = torch.randn(8, 8, 3)
    out = HWC2CHW()(x)
    assert out.shape == (3, 8, 8)


def test_hwc2chw_invalid_dim():
    with pytest.raises(ValueError, match="3-D tensor"):
        HWC2CHW()(torch.zeros(2, 3, 8, 8))


# ---------- Magnitude / component extraction ----------


def test_log_amplitude_preserve_phase():
    x = torch.randn(3, 4, dtype=torch.cfloat) + 0.1
    out = LogAmplitude(scale=2.0, preserve_phase=True)(x)
    assert out.is_complex()
    # Phase preserved
    torch.testing.assert_close(out.angle(), x.angle(), atol=1e-5, rtol=1e-5)


def test_log_amplitude_magnitude_only():
    x = torch.randn(3, 4, dtype=torch.cfloat)
    out = LogAmplitude(preserve_phase=False)(x)
    assert not out.is_complex()


def test_log_amplitude_real_input():
    x = torch.randn(3, 4)
    out = LogAmplitude()(x)
    # Real input -> still real (preserve_phase requires complex)
    assert not out.is_complex()


def test_amplitude():
    x = torch.randn(3, 4, dtype=torch.cfloat)
    torch.testing.assert_close(Amplitude()(x), x.abs())


def test_to_real_complex_and_real():
    x = torch.randn(3, 4, dtype=torch.cfloat)
    torch.testing.assert_close(ToReal()(x), x.real)
    x_r = torch.randn(3, 4)
    torch.testing.assert_close(ToReal()(x_r), x_r)


def test_to_imaginary_complex_and_real():
    x = torch.randn(3, 4, dtype=torch.cfloat)
    torch.testing.assert_close(ToImaginary()(x), x.imag)
    x_r = torch.randn(3, 4)
    torch.testing.assert_close(ToImaginary()(x_r), torch.zeros_like(x_r))


def test_real_imaginary_stack():
    x = torch.randn(2, 8, 8, dtype=torch.cfloat)
    out = RealImaginary()(x)
    assert out.shape == (4, 8, 8)


def test_real_imaginary_real_passthrough():
    x = torch.randn(2, 8, 8)
    out = RealImaginary()(x)
    torch.testing.assert_close(out, x)


# ---------- Normalize ----------


def test_normalize_forward():
    mean = torch.zeros(3, dtype=torch.cfloat)
    cov = torch.eye(2).unsqueeze(0).expand(3, 2, 2).clone()
    norm = Normalize(mean=mean, covariance=cov)
    x = torch.randn(3, 4, 4, dtype=torch.cfloat)
    out = norm(x)
    assert out.shape == x.shape


def test_normalize_invalid_covariance_shape():
    with pytest.raises(ValueError, match="covariance must have shape"):
        Normalize(
            mean=torch.zeros(3, dtype=torch.cfloat), covariance=torch.zeros(3, 2, 3)
        )


def test_normalize_wrong_channel_dim():
    mean = torch.zeros(3, dtype=torch.cfloat)
    cov = torch.eye(2).unsqueeze(0).expand(3, 2, 2).clone()
    norm = Normalize(mean=mean, covariance=cov)
    with pytest.raises(ValueError, match="channel dim"):
        norm(torch.randn(5, 4, 4, dtype=torch.cfloat))


# ---------- RandomPhase ----------


def test_random_phase_preserves_magnitude():
    x = torch.randn(3, 4, dtype=torch.cfloat)
    out = RandomPhase()(x)
    torch.testing.assert_close(out.abs(), x.abs(), atol=1e-5, rtol=1e-5)


def test_random_phase_centered():
    x = torch.randn(3, 4, dtype=torch.cfloat)
    out = RandomPhase(centered=True)(x)
    torch.testing.assert_close(out.abs(), x.abs(), atol=1e-5, rtol=1e-5)


def test_random_phase_real_input_casts():
    x = torch.randn(3, 4)
    out = RandomPhase()(x)
    assert out.is_complex()


# ---------- Spatial ----------


def test_pad_if_needed_smaller():
    x = torch.randn(3, 4, 4, dtype=torch.cfloat)
    out = PadIfNeeded(min_h=8, min_w=8)(x)
    assert out.shape == (3, 8, 8)


def test_pad_if_needed_already_large_returns_input():
    x = torch.randn(3, 8, 8, dtype=torch.cfloat)
    out = PadIfNeeded(min_h=4, min_w=4)(x)
    assert out.shape == (3, 8, 8)


def test_pad_if_needed_real():
    x = torch.randn(3, 4, 4)
    out = PadIfNeeded(min_h=8, min_w=8)(x)
    assert out.shape == (3, 8, 8)


def test_pad_if_needed_invalid_dim_raises():
    """_check_chw rejects 1-D or 2-D inputs."""
    with pytest.raises(ValueError, match="C, H, W"):
        PadIfNeeded(min_h=4, min_w=4)(torch.zeros(8))


def test_center_crop():
    x = torch.randn(3, 8, 8, dtype=torch.cfloat)
    out = CenterCrop(4, 4)(x)
    assert out.shape == (3, 4, 4)


def test_center_crop_too_small_raises():
    with pytest.raises(ValueError, match="larger than input"):
        CenterCrop(16, 16)(torch.randn(3, 4, 4))


def test_spatial_resize_complex_3d():
    x = torch.randn(2, 4, 4, dtype=torch.cfloat)
    out = SpatialResize(8, 8)(x)
    assert out.shape == (2, 8, 8)


def test_spatial_resize_complex_4d():
    x = torch.randn(1, 2, 4, 4, dtype=torch.cfloat)
    out = SpatialResize(8, 8)(x)
    assert out.shape == (1, 2, 8, 8)


def test_spatial_resize_real():
    x = torch.randn(2, 4, 4)
    out = SpatialResize(8, 8)(x)
    assert out.shape == (2, 8, 8)


# ---------- Spectral ----------


def test_fft_ifft_round_trip():
    x = torch.randn(1, 2, 8, 8, dtype=torch.cfloat)
    out = IFFT2()(FFT2()(x))
    torch.testing.assert_close(out, x, atol=1e-5, rtol=1e-5)


def test_fft_resize_downsize():
    x = torch.randn(1, 2, 16, 16, dtype=torch.cfloat)
    out = FFTResize(8, 8)(x)
    assert out.shape == (1, 2, 8, 8)


def test_fft_resize_upsize():
    x = torch.randn(1, 2, 8, 8, dtype=torch.cfloat)
    out = FFTResize(16, 16)(x)
    assert out.shape == (1, 2, 16, 16)


def test_fft_resize_no_energy_preserve():
    x = torch.randn(1, 2, 8, 8, dtype=torch.cfloat)
    out = FFTResize(16, 16, energy_preserving=False)(x)
    assert out.shape == (1, 2, 16, 16)


def test_fft_resize_real_input():
    x = torch.randn(1, 2, 8, 8)
    out = FFTResize(4, 4)(x)
    assert out.shape == (1, 2, 4, 4)


# ---------- PolSAR ----------


@pytest.mark.parametrize(
    "in_c, out_c",
    [
        (1, 1),
        (2, 1),
        (2, 2),
        (3, 1),
        (3, 2),
        (3, 3),
        (4, 1),
        (4, 2),
        (4, 3),
        (4, 4),
    ],
)
def test_polsar_combinations(in_c, out_c):
    x = torch.randn(in_c, 8, 8, dtype=torch.cfloat)
    out = PolSAR(out_channels=out_c)(x)
    assert out.shape[0] == out_c


def test_polsar_out_channels_invalid():
    with pytest.raises(ValueError, match="must be in"):
        PolSAR(out_channels=5)


def test_polsar_too_few_dims_raises():
    with pytest.raises(ValueError, match="at least 3 dims"):
        PolSAR(out_channels=1)(torch.randn(8))


def test_polsar_invalid_combos():
    with pytest.raises(ValueError, match="single-channel"):
        PolSAR(out_channels=2)(torch.randn(1, 8, 8, dtype=torch.cfloat))
    with pytest.raises(ValueError, match="2-channel"):
        PolSAR(out_channels=3)(torch.randn(2, 8, 8, dtype=torch.cfloat))
    with pytest.raises(ValueError, match="3-channel"):
        PolSAR(out_channels=4)(torch.randn(3, 8, 8, dtype=torch.cfloat))


def test_polsar_unsupported_channel_count():
    with pytest.raises(ValueError, match="unsupported input channel"):
        PolSAR(out_channels=1)(torch.randn(5, 8, 8, dtype=torch.cfloat))
