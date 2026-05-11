r"""
Weight Initialization for Complex Tensors
=========================================

Drop-in complex-valued analogues of :mod:`torch.nn.init`. PyTorch's built-in
initializers were designed for real tensors and produce the wrong variance
when applied directly to a complex parameter (each part is treated as
independent of the other, so :math:`\mathrm{Var}(|w|^2)` is too large by a
factor of 2). The functions in this module correct for that.

All functions mutate ``tensor`` in place and return it, mirroring
:mod:`torch.nn.init`.

Functions
---------

- :func:`kaiming_normal_`, :func:`kaiming_uniform_` — He (Kaiming, 2015).
- :func:`xavier_normal_`, :func:`xavier_uniform_` — Glorot (2010).
- :func:`trabelsi_standard_` — polar Rayleigh-uniform initializer from
  Trabelsi et al. (2018) "Deep Complex Networks".
- :func:`trabelsi_independent_` — semi-unitary (orthogonal) complex
  initializer from the same paper.
"""

import math
from typing import Tuple

import torch

__all__ = [
    "kaiming_normal_",
    "kaiming_uniform_",
    "xavier_normal_",
    "xavier_uniform_",
    "trabelsi_standard_",
    "trabelsi_independent_",
]


def _get_fans(tensor: torch.Tensor) -> Tuple[int, int]:
    """Compute fan-in / fan-out for a complex tensor (same as the real form)."""
    if tensor.dim() < 2:
        # Linear bias or 1-D weight: treat as (fan_in,), fan_out = 1.
        fan_in = tensor.numel()
        fan_out = tensor.numel()
        return fan_in, fan_out
    fan_in = (
        tensor.size(1) * tensor[0][0].numel() if tensor.dim() > 2 else tensor.size(1)
    )
    fan_out = (
        tensor.size(0) * tensor[0][0].numel() if tensor.dim() > 2 else tensor.size(0)
    )
    return fan_in, fan_out


def _calculate_gain(nonlinearity: str, a: float = 0.0) -> float:
    """Mirror :func:`torch.nn.init.calculate_gain` for a small set of activations."""
    if nonlinearity in ("linear", "conv1d", "conv2d", "conv3d", "sigmoid"):
        return 1.0
    if nonlinearity == "tanh":
        return 5.0 / 3.0
    if nonlinearity == "relu":
        return math.sqrt(2.0)
    if nonlinearity == "leaky_relu":
        return math.sqrt(2.0 / (1.0 + a * a))
    if nonlinearity == "selu":
        return 3.0 / 4.0
    raise ValueError(f"Unsupported nonlinearity {nonlinearity!r}")


def _check_complex(tensor: torch.Tensor) -> None:
    if not tensor.is_complex():
        raise TypeError(
            f"complextorch.nn.init expects a complex tensor, got dtype={tensor.dtype}"
        )


# ---------------------------------------------------------------------------
# Kaiming / He
# ---------------------------------------------------------------------------


def kaiming_normal_(
    tensor: torch.Tensor,
    a: float = 0.0,
    mode: str = "fan_in",
    nonlinearity: str = "leaky_relu",
) -> torch.Tensor:
    r"""
    Complex Kaiming Normal Initialization
    -------------------------------------

    Draws ``tensor.real`` and ``tensor.imag`` independently from
    :math:`\mathcal{N}(0, \sigma^2)` with
    :math:`\sigma = \text{gain} / \sqrt{2 \cdot \text{fan}}` so that
    :math:`\mathrm{Var}(|w|^2) = 2 \cdot \sigma^2 = \text{gain}^2 / \text{fan}` —
    matching He's target for the complex magnitude.
    """
    _check_complex(tensor)
    fan_in, fan_out = _get_fans(tensor)
    fan = fan_in if mode == "fan_in" else fan_out
    gain = _calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(2.0 * fan)
    with torch.no_grad():
        tensor.real.normal_(0.0, std)
        tensor.imag.normal_(0.0, std)
    return tensor


def kaiming_uniform_(
    tensor: torch.Tensor,
    a: float = 0.0,
    mode: str = "fan_in",
    nonlinearity: str = "leaky_relu",
) -> torch.Tensor:
    r"""Complex Kaiming Uniform Initialization. See :func:`kaiming_normal_`."""
    _check_complex(tensor)
    fan_in, fan_out = _get_fans(tensor)
    fan = fan_in if mode == "fan_in" else fan_out
    gain = _calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(2.0 * fan)
    bound = math.sqrt(3.0) * std  # uniform[-bound, bound] has std = bound/sqrt(3)
    with torch.no_grad():
        tensor.real.uniform_(-bound, bound)
        tensor.imag.uniform_(-bound, bound)
    return tensor


# ---------------------------------------------------------------------------
# Xavier / Glorot
# ---------------------------------------------------------------------------


def xavier_normal_(tensor: torch.Tensor, gain: float = 1.0) -> torch.Tensor:
    r"""
    Complex Xavier Normal Initialization
    ------------------------------------

    Draws each part from :math:`\mathcal{N}(0, \sigma^2)` with
    :math:`\sigma = \text{gain} / \sqrt{\text{fan\_in} + \text{fan\_out}}`
    so that :math:`\mathrm{Var}(|w|^2) = 2 \sigma^2 = 2 \cdot \text{gain}^2 / (\text{fan\_in} + \text{fan\_out})`.
    """
    _check_complex(tensor)
    fan_in, fan_out = _get_fans(tensor)
    std = gain / math.sqrt(fan_in + fan_out)
    with torch.no_grad():
        tensor.real.normal_(0.0, std)
        tensor.imag.normal_(0.0, std)
    return tensor


def xavier_uniform_(tensor: torch.Tensor, gain: float = 1.0) -> torch.Tensor:
    r"""Complex Xavier Uniform Initialization. See :func:`xavier_normal_`."""
    _check_complex(tensor)
    fan_in, fan_out = _get_fans(tensor)
    std = gain / math.sqrt(fan_in + fan_out)
    bound = math.sqrt(3.0) * std
    with torch.no_grad():
        tensor.real.uniform_(-bound, bound)
        tensor.imag.uniform_(-bound, bound)
    return tensor


# ---------------------------------------------------------------------------
# Trabelsi (Deep Complex Networks, 2018)
# ---------------------------------------------------------------------------


def trabelsi_standard_(tensor: torch.Tensor, kind: str = "glorot") -> torch.Tensor:
    r"""
    Trabelsi Polar (Rayleigh-Uniform) Initializer
    ---------------------------------------------

    Polar parameterization from Trabelsi et al. (2018) "Deep Complex Networks".

    The magnitude :math:`|w|` is drawn from a Rayleigh distribution with scale
    :math:`\sigma` and the phase :math:`\arg w` is drawn uniformly from
    :math:`[-\pi, \pi]`:

    .. math::

        |w| \sim \mathrm{Rayleigh}(\sigma), \qquad
        \arg w \sim \mathcal{U}[-\pi, \pi], \qquad
        w = |w| \cdot e^{j \arg w}.

    With ``kind='glorot'``, :math:`\sigma = 1 / \sqrt{\text{fan\_in} + \text{fan\_out}}`;
    with ``kind='he'``, :math:`\sigma = 1 / \sqrt{\text{fan\_in}}`.
    """
    _check_complex(tensor)
    fan_in, fan_out = _get_fans(tensor)
    if kind in ("glorot", "xavier"):
        sigma = 1.0 / math.sqrt(fan_in + fan_out)
    elif kind in ("he", "kaiming"):
        sigma = 1.0 / math.sqrt(fan_in)
    else:
        raise ValueError(
            f"Unknown kind {kind!r}; expected 'glorot'/'xavier' or 'he'/'kaiming'"
        )
    # Rayleigh(sigma) samples: |w| = sigma * sqrt(-2 ln U) for U ~ Uniform(0, 1].
    with torch.no_grad():
        u = torch.empty_like(tensor.real).uniform_(1e-12, 1.0)
        magnitude = sigma * torch.sqrt(-2.0 * torch.log(u))
        phase = torch.empty_like(tensor.real).uniform_(-math.pi, math.pi)
        tensor.real.copy_(magnitude * torch.cos(phase))
        tensor.imag.copy_(magnitude * torch.sin(phase))
    return tensor


def trabelsi_independent_(tensor: torch.Tensor, kind: str = "glorot") -> torch.Tensor:
    r"""
    Trabelsi Semi-Unitary (Independent) Initializer
    -----------------------------------------------

    Complex orthogonal init from Trabelsi et al. (2018).

    Generates a random complex matrix of the same flat shape and replaces its
    singular values with a constant via SVD, yielding a semi-unitary weight
    satisfying :math:`W^* W = c \cdot I` (or :math:`W W^* = c \cdot I` for
    wide matrices). The constant :math:`c` is chosen to match either the
    Glorot or He variance target.
    """
    _check_complex(tensor)
    if tensor.dim() < 2:
        raise ValueError("trabelsi_independent_ requires a tensor of at least 2 dims")

    fan_in, fan_out = _get_fans(tensor)
    if kind in ("glorot", "xavier"):
        scale = 1.0 / math.sqrt(fan_in + fan_out)
    elif kind in ("he", "kaiming"):
        scale = 1.0 / math.sqrt(fan_in)
    else:
        raise ValueError(f"Unknown kind {kind!r}")

    # Flatten to (out, in_total) for SVD.
    out_dim = tensor.size(0)
    in_total = tensor.numel() // out_dim
    rows, cols = out_dim, in_total

    # Draw a random complex matrix and take its (truncated) SVD.
    rng = torch.empty(rows, cols, dtype=tensor.dtype, device=tensor.device)
    with torch.no_grad():
        rng.real.normal_(0.0, 1.0)
        rng.imag.normal_(0.0, 1.0)
        u, _, vh = torch.linalg.svd(rng, full_matrices=False)
        # Smallest dim k = min(rows, cols); semi-unitary product u @ vh has shape (rows, cols).
        w = u @ vh
        w = w * scale
        tensor.copy_(w.reshape(tensor.shape))
    return tensor
