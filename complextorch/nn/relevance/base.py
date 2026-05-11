r"""
BaseARD + Module-Walking Helpers
================================

Adapted from :mod:`cplxmodule.nn.relevance.base`.
"""

from collections.abc import Iterator

import torch

__all__ = [
    "BaseARD",
    "compute_ard_masks",
    "named_penalties",
    "named_relevance",
    "penalties",
]


class BaseARD(torch.nn.Module):
    r"""
    Abstract base for variational-dropout / automatic-relevance-determination
    layers. Subclasses provide:

    - ``.penalty`` (property): differentiable KL divergence to add to the loss.
    - ``.relevance(threshold=...)`` (method): a binary mask of relevant weights.
    """

    @property
    def penalty(self) -> torch.Tensor:
        raise NotImplementedError("Subclasses must compute their own penalty.")

    def relevance(self, **kwargs) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement `.relevance`.")


def named_penalties(
    module: torch.nn.Module,
    reduction: str = "sum",
    prefix: str = "",
) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield ``(name, penalty)`` for every :class:`BaseARD` submodule."""
    if reduction is not None and reduction not in ("mean", "sum"):
        raise ValueError(f"reduction must be 'mean', 'sum', or None; got {reduction!r}")
    for name, mod in module.named_modules(prefix=prefix):
        if isinstance(mod, BaseARD):
            p = mod.penalty
            if reduction == "sum":
                p = p.sum()
            elif reduction == "mean":
                p = p.mean()
            yield name, p


def penalties(
    module: torch.nn.Module, reduction: str = "sum"
) -> Iterator[torch.Tensor]:
    """Yield just the penalty tensors. See :func:`named_penalties`."""
    for _, p in named_penalties(module, reduction=reduction):
        yield p


def named_relevance(
    module: torch.nn.Module, *, prefix: str = "", **kwargs
) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield ``(name, mask)`` for every :class:`BaseARD` submodule."""
    for name, mod in module.named_modules(prefix=prefix):
        if isinstance(mod, BaseARD):
            yield name, mod.relevance(**kwargs).detach()


def compute_ard_masks(module: torch.nn.Module, *, prefix: str = "", **kwargs) -> dict:
    r"""
    Build a ``{name + '.mask': mask}`` dict suitable for
    :func:`complextorch.nn.masked.deploy_masks`.
    """
    if not isinstance(module, torch.nn.Module):
        return {}
    out = {}
    for name, mask in named_relevance(module, prefix=prefix, **kwargs):
        key = (name + "." if name else "") + "mask"
        out[key] = mask
    return out
