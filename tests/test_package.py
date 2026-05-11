"""Top-level package smoke tests."""

from __future__ import annotations

import re

import complextorch


def test_version_is_semver():
    assert re.match(r"^\d+\.\d+\.\d+", complextorch.__version__)


def test_public_subpackages_importable():
    assert complextorch.nn is not None
    assert complextorch.signal is not None
    assert complextorch.transforms is not None
    assert complextorch.datasets is not None
    assert complextorch.models is not None


def test_author_metadata():
    assert isinstance(complextorch.__author__, str)
    assert complextorch.__author__
