"""Shared pytest fixtures for the complextorch test suite."""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch


@pytest.fixture(autouse=True)
def _seed_everything():
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    yield


@pytest.fixture
def device() -> torch.device:
    return torch.device("cpu")


@pytest.fixture
def cplx():
    def _make(
        *shape: int, dtype: torch.dtype = torch.cfloat, requires_grad: bool = False
    ) -> torch.Tensor:
        t = torch.randn(*shape, dtype=dtype)
        if requires_grad:
            t.requires_grad_(True)
        return t

    return _make
