r"""
Complex Diagonal State-Space Models (S4D / DSS / selective Mamba)
=================================================================

Linear-time sequence models whose core is a **diagonal complex** state-space
model (SSM). The complex-diagonal state is exactly what gives structured SSMs
their strong performance on perceptual / signal modalities, which makes them a
natural fit for a complex-valued library and for the long 1-D signals this
package targets.

A single-input single-output diagonal SSM with state :math:`x \in \mathbb{C}^N`
evolves as

.. math::

    x'(t) = A\, x(t) + B\, u(t), \qquad y(t) = C\, x(t) + D\, u(t),

with diagonal :math:`A \in \mathbb{C}^{N}`. Discretising with step
:math:`\Delta` (zero-order hold) gives :math:`\bar{A} = e^{\Delta A}`,
:math:`\bar{B} = (\bar{A}-1)A^{-1}B`, and a causal convolution kernel

.. math::

    \bar{K}_\ell = \sum_{n} C_n\, \bar{A}_n^{\ell}\, \bar{B}_n, \qquad
    y = u * \bar{K} + D\,u .

Training uses an FFT long-convolution with this kernel; :meth:`SSMBase.recurrence`
provides the mathematically-equivalent recurrent rollout (used for exact
inference and verification).

References:

    - **Gu, Goel, Ré. Efficiently Modeling Long Sequences with Structured State
      Spaces (S4).** https://arxiv.org/abs/2111.00396
    - **Gu et al. On the Parameterization and Initialization of Diagonal State
      Space Models (S4D).** https://arxiv.org/abs/2206.11893
    - **Gupta, Gu, Berant. Diagonal State Spaces (DSS).**
      https://arxiv.org/abs/2203.14343
    - **Gu, Dao. Mamba: Linear-Time Sequence Modeling with Selective State
      Spaces.** https://arxiv.org/abs/2312.00752
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from complextorch.nn.modules.activation.split_type_A import CGELU
from complextorch.nn.modules.linear import Linear
from complextorch.nn.modules.rmsnorm import RMSNorm

__all__ = ["DSS", "S4D", "MambaBlock", "S4DBlock"]


class SSMBase(nn.Module):
    r"""
    Diagonal Complex State-Space Model (base, S4D-style)
    ----------------------------------------------------

    Operates on complex sequences of shape ``(B, L, H)`` (batch, length,
    channels). Each of the ``H`` channels owns an independent diagonal SSM with
    ``state_size`` complex states.

    The diagonal matrix is parameterised so its real part is always negative
    (stable): :math:`A = -e^{a_r} + j\,a_i`. ``B`` and ``C`` are complex
    parameters; the per-channel step ``dt`` and skip ``D`` are real.

    Initialisation follows S4D-Lin: :math:`A_n = -\tfrac12 + j\pi n`.

    Args:
        channels: number of independent channels ``H``.
        state_size: state dimension ``N`` per channel.
        dt_min, dt_max: log-uniform range for the initial step sizes.
    """

    def __init__(
        self,
        channels: int,
        state_size: int = 64,
        dt_min: float = 1e-3,
        dt_max: float = 1e-1,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.state_size = state_size

        # A = -exp(log_neg_A_real) + j * A_imag  (real part stays negative).
        self.log_neg_A_real = nn.Parameter(
            torch.full((channels, state_size), math.log(0.5))
        )
        a_imag = math.pi * torch.arange(state_size, dtype=torch.float32)
        self.A_imag = nn.Parameter(a_imag.expand(channels, state_size).clone())

        self.B = nn.Parameter(torch.ones(channels, state_size, dtype=torch.cfloat))
        self.C = nn.Parameter(torch.randn(channels, state_size, dtype=torch.cfloat))
        self.D = nn.Parameter(torch.ones(channels))

        log_dt = torch.rand(channels) * (
            math.log(dt_max) - math.log(dt_min)
        ) + math.log(dt_min)
        self.log_dt = nn.Parameter(log_dt)

    # -- parameter views -----------------------------------------------------

    def _A(self) -> torch.Tensor:
        return torch.complex(-torch.exp(self.log_neg_A_real), self.A_imag)

    def _dtA(self) -> torch.Tensor:
        """Discretisation exponent ``dt * A`` of shape ``(H, N)``."""
        return torch.exp(self.log_dt).unsqueeze(-1) * self._A()

    def _discretize(self, length: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(A_bar, B_bar)``; zero-order hold (length-independent)."""
        dtA = self._dtA()
        A_bar = torch.exp(dtA)
        B_bar = (A_bar - 1.0) / self._A() * self.B
        return A_bar, B_bar

    # -- kernel / forward ----------------------------------------------------

    def _kernel(self, length: int) -> torch.Tensor:
        """Causal convolution kernel ``K`` of shape ``(H, L)``."""
        _, B_bar = self._discretize(length)
        dtA = self._dtA()  # log of A_bar
        powers = torch.arange(length, device=dtA.device, dtype=dtA.real.dtype)
        vander = torch.exp(dtA.unsqueeze(-1) * powers)  # (H, N, L)
        return torch.einsum("hn,hnl->hl", self.C * B_bar, vander)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        r"""FFT long-convolution of the input with the SSM kernel.

        Args:
            input (torch.Tensor): complex ``(B, L, H)`` sequence.

        Returns:
            torch.Tensor: complex ``(B, L, H)`` sequence.
        """
        length = input.shape[1]
        kernel = self._kernel(length)  # (H, L)
        nfft = 2 * length
        kf = torch.fft.fft(kernel, n=nfft, dim=-1)  # (H, nfft)
        uf = torch.fft.fft(input, n=nfft, dim=1)  # (B, nfft, H)
        yf = uf * kf.transpose(0, 1).unsqueeze(0)  # (B, nfft, H)
        y = torch.fft.ifft(yf, dim=1)[:, :length, :]
        return y + self.D * input

    def recurrence(self, input: torch.Tensor) -> torch.Tensor:
        r"""Exact recurrent rollout; equals :meth:`forward` up to FFT error.

        Useful for streaming inference and as a reference implementation.

        Args:
            input (torch.Tensor): complex ``(B, L, H)`` sequence.

        Returns:
            torch.Tensor: complex ``(B, L, H)`` sequence.
        """
        batch, length, _ = input.shape
        A_bar, B_bar = self._discretize(length)
        state = torch.zeros(
            batch,
            self.channels,
            self.state_size,
            dtype=torch.cfloat,
            device=input.device,
        )
        outputs = []
        for t in range(length):
            u_t = input[:, t, :]  # (B, H)
            state = A_bar * state + B_bar * u_t.unsqueeze(-1)
            y_t = (self.C * state).sum(-1) + self.D * u_t
            outputs.append(y_t)
        return torch.stack(outputs, dim=1)

    def extra_repr(self) -> str:
        return f"channels={self.channels}, state_size={self.state_size}"


class S4D(SSMBase):
    r"""
    S4D -- Diagonal Complex State-Space Model
    -----------------------------------------

    Zero-order-hold diagonal SSM (Gu et al., https://arxiv.org/abs/2206.11893).
    See :class:`SSMBase` for the full description and arguments.
    """


class DSS(SSMBase):
    r"""
    DSS -- Diagonal State Space with normalised kernel
    --------------------------------------------------

    Variant of :class:`S4D` using the Diagonal-State-Space normalisation
    (Gupta et al., https://arxiv.org/abs/2203.14343): the input matrix is
    rescaled by :math:`\Delta A / (e^{L \Delta A} - 1)`, which bounds the kernel
    over the sequence length :math:`L` regardless of the sign of
    :math:`\Re(A)`. All other behaviour (kernel construction, FFT convolution,
    recurrence) is inherited unchanged, so :meth:`forward` and
    :meth:`recurrence` remain mathematically equivalent.
    """

    def _discretize(self, length: int) -> tuple[torch.Tensor, torch.Tensor]:
        dtA = self._dtA()
        A_bar = torch.exp(dtA)
        B_bar = self.B * dtA / (torch.exp(length * dtA) - 1.0)
        return A_bar, B_bar


class S4DBlock(nn.Module):
    r"""
    Residual S4D Block
    ------------------

    Stackable block: pre-:class:`~complextorch.nn.RMSNorm` -> diagonal SSM ->
    complex GELU -> :class:`~complextorch.nn.Linear`, with a residual
    connection. Operates on ``(B, L, H)`` complex sequences.

    Args:
        channels: number of channels ``H``.
        state_size: SSM state dimension.
        variant: ``'s4d'`` (default) or ``'dss'``.
        layer_norm_eps: epsilon for the RMSNorm.
    """

    def __init__(
        self,
        channels: int,
        state_size: int = 64,
        variant: str = "s4d",
        layer_norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        if variant not in ("s4d", "dss"):
            raise ValueError(f"variant must be 's4d' or 'dss'; got {variant!r}")
        self.norm = RMSNorm(channels, eps=layer_norm_eps)
        ssm_cls = S4D if variant == "s4d" else DSS
        self.ssm = ssm_cls(channels, state_size)
        self.act = CGELU()
        self.out = Linear(channels, channels)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        z = self.norm(input)
        z = self.ssm(z)
        z = self.out(self.act(z))
        return input + z


class MambaBlock(nn.Module):
    r"""
    Selective (Mamba-style) Complex State-Space Block
    -------------------------------------------------

    Complex adaptation of the Mamba S6 selective SSM
    (https://arxiv.org/abs/2312.00752): the input matrix ``B``, output matrix
    ``C`` and step ``dt`` are **input-dependent**, so the model can selectively
    propagate or forget information. Because the dynamics are time-varying there
    is no global convolution kernel; the state is advanced with a sequential
    selective scan (pure-torch; no custom kernel).

    Operates on ``(B, L, channels)`` complex sequences.

    Args:
        channels: model dimension.
        state_size: SSM state dimension ``N``.
        expand: inner-dimension expansion factor.
        dt_rank: rank of the low-rank ``dt`` projection (defaults to
            ``max(1, channels // 16)``).
    """

    def __init__(
        self,
        channels: int,
        state_size: int = 16,
        expand: int = 2,
        dt_rank: int | None = None,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.state_size = state_size
        self.d_inner = expand * channels
        self.dt_rank = dt_rank if dt_rank is not None else max(1, channels // 16)

        self.in_proj = Linear(channels, self.d_inner)
        self.x_proj = Linear(self.d_inner, self.dt_rank + 2 * state_size)
        self.dt_proj = Linear(self.dt_rank, self.d_inner)
        self.out_proj = Linear(self.d_inner, channels)

        # Static diagonal A (S4D-Lin init), shared across the scan.
        self.log_neg_A_real = nn.Parameter(
            torch.full((self.d_inner, state_size), math.log(0.5))
        )
        a_imag = math.pi * torch.arange(state_size, dtype=torch.float32)
        self.A_imag = nn.Parameter(a_imag.expand(self.d_inner, state_size).clone())
        self.D = nn.Parameter(torch.ones(self.d_inner))

    def _A(self) -> torch.Tensor:
        return torch.complex(-torch.exp(self.log_neg_A_real), self.A_imag)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        r"""Selective scan over the sequence.

        Args:
            input (torch.Tensor): complex ``(B, L, channels)`` sequence.

        Returns:
            torch.Tensor: complex ``(B, L, channels)`` sequence.
        """
        batch, length, _ = input.shape
        x = self.in_proj(input)  # (B, L, d_inner)

        proj = self.x_proj(x)  # (B, L, dt_rank + 2N)
        dt_part = proj[..., : self.dt_rank]
        B_t = proj[..., self.dt_rank : self.dt_rank + self.state_size]  # (B, L, N)
        C_t = proj[..., self.dt_rank + self.state_size :]  # (B, L, N)
        # dt is real & positive (selective gate); derive it from the real part.
        dt = F.softplus(self.dt_proj(dt_part).real)  # (B, L, d_inner)

        A = self._A()  # (d_inner, N)
        state = torch.zeros(
            batch,
            self.d_inner,
            self.state_size,
            dtype=torch.cfloat,
            device=input.device,
        )
        outputs = []
        for t in range(length):
            dt_t = dt[:, t].unsqueeze(-1)  # (B, d_inner, 1)
            A_bar = torch.exp(dt_t * A)  # (B, d_inner, N)
            B_bar = dt_t * B_t[:, t].unsqueeze(1)  # (B, d_inner, N)
            state = A_bar * state + B_bar * x[:, t].unsqueeze(-1)
            y_t = (C_t[:, t].unsqueeze(1) * state).sum(-1) + self.D * x[:, t]
            outputs.append(y_t)
        y = torch.stack(outputs, dim=1)  # (B, L, d_inner)
        return self.out_proj(y)

    def extra_repr(self) -> str:
        return (
            f"channels={self.channels}, state_size={self.state_size}, "
            f"d_inner={self.d_inner}, dt_rank={self.dt_rank}"
        )
