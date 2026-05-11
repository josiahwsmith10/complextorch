r"""
Masked / Pruned Layers + Module-Walking Helpers
===============================================

Fixed-sparsity-pattern complex layers and helpers for managing their masks
across a whole network.
"""

from typing import Dict, Iterator, Tuple

import torch

from complextorch.nn.masked.base import BaseMasked, MaskedWeightMixin
from complextorch.nn.masked.conv import Conv1dMasked, Conv2dMasked, Conv3dMasked
from complextorch.nn.masked.linear import BilinearMasked, LinearMasked

__all__ = [
    "BaseMasked",
    "MaskedWeightMixin",
    "LinearMasked",
    "BilinearMasked",
    "Conv1dMasked",
    "Conv2dMasked",
    "Conv3dMasked",
    "deploy_masks",
    "binarize_masks",
    "is_sparse",
    "named_masks",
]


def deploy_masks(
    model: torch.nn.Module,
    state_dict: Dict[str, torch.Tensor],
    *,
    strict: bool = True,
) -> torch.nn.Module:
    r"""
    Load a ``{name: mask}`` dict into the matching :class:`BaseMasked`
    submodules of ``model``.

    Keys in ``state_dict`` are interpreted as fully-qualified module names
    (e.g. ``"encoder.layer1.linear.mask"`` or just ``"linear.mask"``). Any
    key ending in ``".mask"`` is matched to the corresponding submodule's
    ``mask`` buffer.
    """
    for full_key, value in state_dict.items():
        if not full_key.endswith("mask"):
            continue
        mod_path = (
            full_key[: -len(".mask")]
            if full_key.endswith(".mask")
            else full_key[: -len("mask")]
        )
        # Walk to the target module.
        mod = model
        if mod_path:
            for part in mod_path.split("."):
                mod = getattr(mod, part, None)
                if mod is None:
                    break
        if isinstance(mod, BaseMasked):
            mod.mask_(value)
        elif strict:
            raise KeyError(f"deploy_masks: no BaseMasked submodule at {mod_path!r}")
    return model


def binarize_masks(model: torch.nn.Module) -> torch.nn.Module:
    r"""In-place binarize every mask attached to a :class:`BaseMasked` submodule."""
    for mod in model.modules():
        if isinstance(mod, BaseMasked) and mod.is_sparse:
            mod.mask_((mod.mask != 0).to(mod.mask.dtype))
    return model


def is_sparse(layer: torch.nn.Module) -> bool:
    """``True`` if ``layer`` is a :class:`BaseMasked` with a mask set."""
    return isinstance(layer, BaseMasked) and layer.is_sparse


def named_masks(
    model: torch.nn.Module,
) -> Iterator[Tuple[str, torch.Tensor]]:
    """Yield ``(qualified_name, mask)`` for each currently-set mask."""
    for name, mod in model.named_modules():
        if isinstance(mod, BaseMasked) and mod.is_sparse:
            yield name, mod.mask
