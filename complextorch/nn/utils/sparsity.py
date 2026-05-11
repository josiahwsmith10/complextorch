r"""
Sparsity Statistics Helpers
===========================

Walk a module tree and report sparsity stats for layers that subclass
:class:`SparsityStats` (typically a :mod:`complextorch.nn.masked` or
:mod:`complextorch.nn.relevance` layer).
"""

from typing import Iterator, Tuple

import torch

__all__ = ["SparsityStats", "named_sparsity", "sparsity"]


class SparsityStats(torch.nn.Module):
    r"""
    Mixin for modules that can report ``n_zeros`` per parameter.

    Subclasses must implement :meth:`sparsity(self, **kwargs)`, returning a
    list of ``(param_id, n_dropped)`` tuples where ``param_id`` uniquely
    identifies a parameter (we use ``id(p)``).
    """

    __sparsity_ignore__: tuple = ()

    def sparsity(self, **kwargs):
        raise NotImplementedError(
            "Subclasses of SparsityStats must implement `.sparsity(self, **kwargs)`"
        )


def named_sparsity(
    module: torch.nn.Module, *, prefix: str = "", **kwargs
) -> Iterator[Tuple[str, Tuple[int, int]]]:
    r"""
    Yield ``(param_name, (n_zeros, n_total))`` for each parameter of every
    :class:`SparsityStats` submodule.

    ``kwargs`` are forwarded to :meth:`SparsityStats.sparsity` (typically
    ``threshold=...``).
    """
    # Build a mapping id(param) -> ("module_name.param_name", n_total).
    pid_to_name = {}
    pid_to_total = {}
    for mod_name, mod in module.named_modules(prefix=prefix):
        for p_name, p in mod.named_parameters(recurse=False):
            full = f"{mod_name}.{p_name}" if mod_name else p_name
            pid_to_name[id(p)] = full
            pid_to_total[id(p)] = p.numel()

    seen = set()
    for mod_name, mod in module.named_modules(prefix=prefix):
        if not isinstance(mod, SparsityStats):
            continue
        for pid, n_dropped in mod.sparsity(**kwargs):
            if pid in seen or pid not in pid_to_name:
                continue
            seen.add(pid)
            yield pid_to_name[pid], (int(n_dropped), int(pid_to_total[pid]))


def sparsity(module: torch.nn.Module, **kwargs) -> float:
    """Return the overall sparsity ratio (``n_zeros / n_total``)."""
    total = 0
    zeros = 0
    for _, (n_z, n_t) in named_sparsity(module, **kwargs):
        zeros += n_z
        total += n_t
    return zeros / total if total else 0.0
