r"""Private mask canonicalization shared by the attention modules.

Lives in its own module so both ``attention/__init__.py`` and
``attention/holographic.py`` can import it without a circular import
(``__init__`` imports ``holographic``).
"""

import torch

__all__ = ["_canonical_mask"]


def _canonical_mask(mask: torch.Tensor | None, name: str) -> torch.Tensor | None:
    """Convert a bool mask to an additive float mask (0 keep / -inf drop);
    pass float masks through; reject other dtypes."""
    if mask is None:
        return None
    if mask.dtype == torch.bool:
        return torch.zeros_like(mask, dtype=torch.float32).masked_fill_(
            mask, float("-inf")
        )
    if mask.is_floating_point():
        return mask
    raise ValueError(
        f"{name} must be a bool or floating-point tensor, got {mask.dtype}"
    )
