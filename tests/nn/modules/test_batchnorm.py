"""Tests for complex BatchNorm{1,2,3}d, NaiveBatchNorm{1,2,3}d, MagBatchNorm{1,2,3}d."""

from __future__ import annotations

import pytest
import torch

from complextorch.nn.modules.batchnorm import (
    BatchNorm1d,
    BatchNorm2d,
    BatchNorm3d,
    MagBatchNorm1d,
    MagBatchNorm2d,
    MagBatchNorm3d,
    NaiveBatchNorm1d,
    NaiveBatchNorm2d,
    NaiveBatchNorm3d,
)

# ---------- Trabelsi BN (2x2 whitening) ----------


@pytest.mark.parametrize(
    ("cls", "shape"),
    [
        (BatchNorm1d, (16, 4, 10)),
        (BatchNorm2d, (16, 4, 8, 8)),
        (BatchNorm3d, (4, 4, 4, 4, 4)),
    ],
)
def test_batchnorm_forward_train_and_eval(cls, shape):
    bn = cls(num_features=4)
    x = torch.randn(*shape, dtype=torch.cfloat)
    bn.train()
    out = bn(x)
    assert out.shape == x.shape
    assert out.is_complex()
    # running stats should now be non-trivial
    assert (bn.running_mean.abs().sum() > 0) or (bn.num_batches_tracked.item() == 1)
    bn.eval()
    out2 = bn(x)
    assert out2.shape == x.shape


def test_batchnorm1d_2d_input():
    """BatchNorm1d accepts both 2D and 3D inputs."""
    bn = BatchNorm1d(num_features=4)
    x = torch.randn(8, 4, dtype=torch.cfloat)
    out = bn(x)
    assert out.shape == (8, 4)


def test_batchnorm1d_invalid_dim():
    bn = BatchNorm1d(4)
    with pytest.raises(ValueError, match="expected 2D or 3D"):
        bn(torch.randn(2, 4, 4, 4, dtype=torch.cfloat))


def test_batchnorm2d_invalid_dim():
    bn = BatchNorm2d(4)
    with pytest.raises(ValueError, match="expected 4D"):
        bn(torch.randn(2, 4, 4, dtype=torch.cfloat))


def test_batchnorm3d_invalid_dim():
    bn = BatchNorm3d(4)
    with pytest.raises(ValueError, match="expected 5D"):
        bn(torch.randn(2, 4, 4, 4, dtype=torch.cfloat))


def test_batchnorm_no_affine_no_track():
    bn = BatchNorm2d(4, affine=False, track_running_stats=False)
    assert bn.weight is None
    assert bn.bias is None
    assert bn.running_mean is None
    x = torch.randn(8, 4, 6, 6, dtype=torch.cfloat)
    out = bn(x)
    assert out.shape == x.shape


def test_batchnorm_momentum_none_uses_cumulative():
    bn = BatchNorm2d(4, momentum=None)
    bn.train()
    x = torch.randn(8, 4, 4, 4, dtype=torch.cfloat)
    bn(x)
    bn(x)
    assert bn.num_batches_tracked.item() == 2


def test_batchnorm_extra_repr():
    s = BatchNorm2d(4).extra_repr()
    assert "4" in s  # the format string starts with {num_features}
    assert "affine=True" in s


def test_batchnorm_reset_running_stats_idempotent_when_untracked():
    bn = BatchNorm1d(4, track_running_stats=False)
    bn.reset_running_stats()  # no-op when not tracking


def test_batchnorm_reset_parameters_no_op_no_affine():
    bn = BatchNorm1d(4, affine=False)
    bn.reset_parameters()  # no-op


# ---------- Naive (split) BN ----------


@pytest.mark.parametrize(
    ("cls", "shape"),
    [
        (NaiveBatchNorm1d, (16, 4, 10)),
        (NaiveBatchNorm2d, (16, 4, 8, 8)),
        (NaiveBatchNorm3d, (4, 4, 4, 4, 4)),
    ],
)
def test_naive_batchnorm_forward(cls, shape):
    bn = cls(num_features=4)
    x = torch.randn(*shape, dtype=torch.cfloat)
    out = bn(x)
    assert out.shape == x.shape
    assert out.is_complex()


def test_naive_batchnorm_extra_repr():
    s = NaiveBatchNorm2d(4).extra_repr()
    assert "num_features=4" in s


# ---------- MagBatchNorm (magnitude-only, equivariant) ----------


@pytest.mark.parametrize(
    ("cls", "shape"),
    [
        (MagBatchNorm1d, (16, 4, 10)),
        (MagBatchNorm2d, (16, 4, 8, 8)),
        (MagBatchNorm3d, (4, 4, 4, 4, 4)),
    ],
)
def test_mag_batchnorm_forward(cls, shape):
    bn = cls(num_features=4)
    x = torch.randn(*shape, dtype=torch.cfloat)
    out = bn(x)
    assert out.shape == x.shape
    assert out.is_complex()


def test_mag_batchnorm_equivariant_under_phase_rotation():
    bn = MagBatchNorm2d(num_features=4)
    bn.eval()
    bn.train()
    _ = bn(torch.randn(8, 4, 6, 6, dtype=torch.cfloat))
    bn.eval()
    x = torch.randn(2, 4, 6, 6, dtype=torch.cfloat) + 0.1
    rotor = torch.polar(torch.tensor(1.0), torch.tensor(0.9))
    y1 = bn(x * rotor)
    y2 = bn(x) * rotor
    torch.testing.assert_close(y1, y2, atol=1e-4, rtol=1e-4)


def test_mag_batchnorm_state_dict_uses_underlying_bn():
    bn = MagBatchNorm2d(num_features=4)
    sd_keys = list(bn.state_dict().keys())
    # The underlying real BN's params should be exposed as bn.weight, bn.bias, etc.
    assert any(k.startswith("bn.") for k in sd_keys)


def test_mag_batchnorm_extra_repr():
    s = MagBatchNorm2d(4).extra_repr()
    assert "num_features=4" in s


def test_mag_batchnorm_real_input_passthrough():
    """Real input goes straight through the underlying real BN (else branch)."""
    bn = MagBatchNorm2d(num_features=4)
    bn.train()
    x = torch.randn(8, 4, 6, 6)
    out = bn(x)
    assert not out.is_complex()
    assert out.shape == x.shape
