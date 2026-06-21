r"""
Complex-Valued Positional Encodings
===================================

Positional information for complex-valued sequence models (e.g.
:class:`complextorch.nn.MultiheadAttention`,
:class:`complextorch.nn.TransformerEncoderLayer`). The native
:class:`complextorch.nn.Transformer` applies **no** positional encoding on its
own, so one of these modules must be used to make attention position-aware.

Three flavours are provided:

- :class:`RotaryEmbedding` -- **relative** position via Rotary Position
  Embedding (RoPE). Each complex feature channel :math:`k` of a token at
  position :math:`n` is multiplied by a unit phasor :math:`e^{j\omega_k n}`.
  Under the Hermitian attention inner product :math:`q\,k^H` the per-channel
  contribution becomes :math:`q_k \overline{k_k}\, e^{j\omega_k (m-n)}`, which
  depends only on the relative offset :math:`m-n`. Apply it to the query and
  key tensors (see :meth:`RotaryEmbedding.rotate_q_k`).
- :class:`SinusoidalPositionalEncoding` -- fixed **absolute** encoding; adds a
  complex sinusoidal phasor bank :math:`e^{j\omega_k n}` to the embeddings.
- :class:`CoPE` -- a lightweight **learnable** complex positional encoding with
  per-channel learnable frequency and phase offset (``2 * dim`` parameters).

RoPE is described in:

    **J. Su, et al. RoFormer: Enhanced Transformer with Rotary Position Embedding.**

        - https://arxiv.org/abs/2104.09864

The complex / phase-modulation reading of RoPE (each rotation is literally a
multiplication by :math:`e^{j\theta}` on a bank of complex oscillators) makes it
a natural fit for a complex-valued library.
"""

import torch
import torch.nn as nn

__all__ = ["CoPE", "RotaryEmbedding", "SinusoidalPositionalEncoding"]


def _inv_freq(dim: int, base: float) -> torch.Tensor:
    """Geometrically-spaced angular frequencies, one per complex channel."""
    return 1.0 / (base ** (torch.arange(0, dim, dtype=torch.float32) / dim))


def _position_angles(inv_freq: torch.Tensor, seq_len: int, offset: int) -> torch.Tensor:
    """Outer product ``pos x inv_freq`` -> angle tensor of shape ``(L, dim)``."""
    pos = torch.arange(
        offset, offset + seq_len, device=inv_freq.device, dtype=inv_freq.dtype
    )
    return torch.outer(pos, inv_freq)


class RotaryEmbedding(nn.Module):
    r"""
    Complex Rotary Position Embedding (RoPE)
    ----------------------------------------

    Multiplies the complex feature at sequence position :math:`n` by the
    position-dependent unit phasor :math:`e^{j\omega_k n}`:

    .. math::

        \tilde{x}_{n,k} = x_{n,k}\, e^{j \omega_k n},
        \qquad \omega_k = \text{base}^{-k/d}.

    Apply it to the query and key tensors *before* the attention dot product.
    Because the attention score uses the Hermitian inner product
    :math:`q\,k^H`, the rotation injected at positions :math:`m` (query) and
    :math:`n` (key) leaves a residual phase :math:`e^{j\omega_k(m-n)}` that
    encodes the **relative** position.

    The last tensor dimension is treated as the (complex) feature dimension and
    must equal ``dim``; the second-to-last dimension is the sequence dimension.
    This matches the per-head attention layout ``(B, n_heads, L, d_k)`` as well
    as a plain ``(B, L, d)`` embedding layout.

    Args:
        dim: number of complex feature channels (size of the last dim).
        base: geometric base for the frequency schedule (RoPE default 10000).
        learnable: if ``True`` the per-channel frequencies are learnable.
    """

    def __init__(
        self, dim: int, base: float = 10000.0, learnable: bool = False
    ) -> None:
        super().__init__()
        self.dim = dim
        self.base = base
        self.learnable = learnable
        inv_freq = _inv_freq(dim, base)
        if learnable:
            self.inv_freq = nn.Parameter(inv_freq)
        else:
            self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, input: torch.Tensor, offset: int = 0) -> torch.Tensor:
        r"""Rotate ``input`` by the RoPE phasors.

        Args:
            input (torch.Tensor): ``(..., L, dim)`` tensor; the last dim is the
                complex feature dim and the second-to-last is the sequence dim.
            offset (int): position of the first token (useful for KV caching).

        Returns:
            torch.Tensor: rotated complex tensor of the same shape.
        """
        if input.shape[-1] != self.dim:
            raise ValueError(
                f"last dim of input ({input.shape[-1]}) must equal dim ({self.dim})"
            )
        angles = _position_angles(self.inv_freq, input.shape[-2], offset)
        rotor = torch.polar(torch.ones_like(angles), angles)
        if not input.is_complex():
            input = input.to(torch.cfloat)
        return input * rotor

    def rotate_q_k(
        self, q: torch.Tensor, k: torch.Tensor, offset: int = 0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply RoPE to both the query and key tensors."""
        return self.forward(q, offset), self.forward(k, offset)

    def extra_repr(self) -> str:
        return f"dim={self.dim}, base={self.base}, learnable={self.learnable}"


class SinusoidalPositionalEncoding(nn.Module):
    r"""
    Complex Sinusoidal Positional Encoding
    --------------------------------------

    Fixed **absolute** positional encoding that adds a bank of complex
    sinusoids to the input embeddings:

    .. math::

        \tilde{x}_{n,k} = x_{n,k} + e^{j \omega_k n},
        \qquad \omega_k = \text{base}^{-k/d}.

    The last tensor dimension is the (complex) feature dimension and must equal
    ``dim``; the second-to-last dimension is the sequence dimension.

    Args:
        dim: number of complex feature channels (size of the last dim).
        base: geometric base for the frequency schedule.
    """

    def __init__(self, dim: int, base: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.base = base
        self.register_buffer("inv_freq", _inv_freq(dim, base), persistent=False)

    def forward(self, input: torch.Tensor, offset: int = 0) -> torch.Tensor:
        r"""Add the complex sinusoidal encoding to ``input``.

        Args:
            input (torch.Tensor): ``(..., L, dim)`` complex tensor.
            offset (int): position of the first token.

        Returns:
            torch.Tensor: ``input`` plus the positional phasor bank.
        """
        if input.shape[-1] != self.dim:
            raise ValueError(
                f"last dim of input ({input.shape[-1]}) must equal dim ({self.dim})"
            )
        angles = _position_angles(self.inv_freq, input.shape[-2], offset)
        pe = torch.polar(torch.ones_like(angles), angles)
        if not input.is_complex():
            input = input.to(torch.cfloat)
        return input + pe

    def extra_repr(self) -> str:
        return f"dim={self.dim}, base={self.base}"


class CoPE(nn.Module):
    r"""
    Lightweight Learnable Complex Positional Encoding
    -------------------------------------------------

    Multiplies the complex feature at position :math:`n` by a learnable rotor

    .. math::

        \tilde{x}_{n,k} = x_{n,k}\, e^{j (\omega_k n + \phi_k)},

    where both the per-channel frequency :math:`\omega_k` and phase offset
    :math:`\phi_k` are learnable (``2 * dim`` parameters total). Unlike
    :class:`RotaryEmbedding` -- which uses a *fixed* frequency schedule and is
    designed to be applied to the query/key tensors for *relative* encoding --
    :class:`CoPE` is a standalone learnable *absolute* encoding applied to the
    sequence embeddings.

    The frequencies are initialised to the RoPE schedule and the phase offsets
    to zero, so an untrained :class:`CoPE` behaves like :class:`RotaryEmbedding`.

    Args:
        dim: number of complex feature channels (size of the last dim).
        base: geometric base used to initialise the frequencies.
    """

    def __init__(self, dim: int, base: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.base = base
        self.omega = nn.Parameter(_inv_freq(dim, base))
        self.phi = nn.Parameter(torch.zeros(dim))

    def forward(self, input: torch.Tensor, offset: int = 0) -> torch.Tensor:
        r"""Rotate ``input`` by the learnable positional rotor.

        Args:
            input (torch.Tensor): ``(..., L, dim)`` complex tensor.
            offset (int): position of the first token.

        Returns:
            torch.Tensor: rotated complex tensor of the same shape.
        """
        if input.shape[-1] != self.dim:
            raise ValueError(
                f"last dim of input ({input.shape[-1]}) must equal dim ({self.dim})"
            )
        angles = _position_angles(self.omega, input.shape[-2], offset) + self.phi
        rotor = torch.polar(torch.ones_like(angles), angles)
        if not input.is_complex():
            input = input.to(torch.cfloat)
        return input * rotor

    def extra_repr(self) -> str:
        return f"dim={self.dim}, base={self.base}"
