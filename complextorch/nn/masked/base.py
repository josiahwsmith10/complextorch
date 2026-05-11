r"""
Masked-Layer Base Classes
=========================

Building blocks for layers that apply a fixed binary mask to their weight at
forward time. Used for inference-time pruning: train with
:mod:`complextorch.nn.relevance`, extract a relevance mask via
:func:`complextorch.nn.relevance.compute_ard_masks`, then load it into a
masked layer with :func:`complextorch.nn.masked.deploy_masks`.

Adapted from :mod:`cplxmodule.nn.masked.base`.
"""

import torch
import torch.nn as nn

from complextorch.nn.utils.sparsity import SparsityStats

__all__ = ["BaseMasked", "MaskedWeightMixin"]


class BaseMasked(nn.Module):
    r"""
    Base for layers with a fixed binary mask buffer applied to ``self.weight``.

    Attributes:
        is_sparse: ``True`` if a mask is currently set.
        mask: a real-valued tensor of the same shape as the parameter, with
            ``0`` marking a dropped weight and ``1`` marking a kept weight.
    """

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("mask", None)

    @property
    def is_sparse(self) -> bool:
        return isinstance(self.mask, torch.Tensor)

    def mask_(self, mask):
        if mask is not None and not isinstance(mask, torch.Tensor):
            raise TypeError(
                f"`mask` must be a Tensor or None, got {type(mask).__name__}"
            )
        if mask is not None:
            mask = mask.detach().to(
                self.weight.device,
                (
                    self.weight.real.dtype
                    if self.weight.is_complex()
                    else self.weight.dtype
                ),
            )
            mask = mask.expand(self.weight.shape).contiguous()
            self.register_buffer("mask", mask)
        elif self.is_sparse and mask is None:
            del self.mask
            self.register_buffer("mask", None)
        return self

    def __setattr__(self, name, value):
        if name != "mask":
            return super().__setattr__(name, value)
        self.mask_(value)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        mask_key = prefix + "mask"
        super()._load_from_state_dict(
            {k: v for k, v in state_dict.items() if k != mask_key},
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        mask_in_missing = mask_key in missing_keys
        if mask_key in state_dict:
            if mask_in_missing:
                missing_keys.remove(mask_key)
            self.mask_(state_dict[mask_key])
        elif strict:
            if not mask_in_missing:
                missing_keys.append(mask_key)
        elif mask_in_missing:
            missing_keys.remove(mask_key)


class MaskedWeightMixin(SparsityStats):
    r"""Provides ``weight_masked`` returning ``self.weight * self.mask``."""

    __sparsity_ignore__ = ()

    @property
    def weight_masked(self) -> torch.Tensor:
        if not getattr(self, "is_sparse", False):
            raise RuntimeError(
                f"`{type(self).__name__}` has no sparsity mask. "
                "Set ``.mask`` or call ``deploy_masks(...)``."
            )
        # Complex weight * real mask broadcasts correctly.
        return self.weight * self.mask

    def sparsity(self, **kwargs):
        weight = self.weight
        if self.is_sparse:
            n_dropped = float((self.mask == 0).sum().item())
        else:
            n_dropped = 0.0
        if weight.is_complex():
            return [(id(weight), n_dropped)]
        return [(id(weight), n_dropped)]
