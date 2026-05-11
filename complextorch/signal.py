r"""
Complex-Aware Signal-Processing Utilities
=========================================

A small set of complex-aware signal helpers that don't fit naturally in
:mod:`complextorch.nn`. Currently:

- :func:`pwelch` — Welch power spectral density (torch port of
  :func:`scipy.signal.welch`). Works on both real and complex inputs and is
  differentiable end-to-end.
"""

import torch

__all__ = ["pwelch"]


def _window_view(x: torch.Tensor, dim: int, size: int, stride: int = 1) -> torch.Tensor:
    """Sliding-window view of ``x`` along ``dim``. Returns a view of shape
    ``(..., n_windows, size, ...)`` (the new window axis replaces ``dim``,
    with the per-window-time axis appended immediately after)."""
    return x.unfold(dim, size, stride)


def pwelch(
    x: torch.Tensor,
    window: int | torch.Tensor = 256,
    fs: float = 1.0,
    scaling: str = "density",
    n_overlap: int | None = None,
    detrend: str = "constant",
    return_onesided: bool | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""
    Welch Power Spectral Density (torch).

    Mirrors the headline arguments of :func:`scipy.signal.welch`. Works on
    real or complex ``x``; for a complex input the returned spectrum is
    two-sided by default (matching scipy).

    Args:
        x: signal tensor; PSD is computed over the **last** dim.
        window: integer window length (uses Hann), or a 1-D tensor with the
            window samples. Defaults to 256 (or the signal length if shorter).
        fs: sampling frequency (in Hz). Defaults to ``1.0`` so that the
            returned ``frequencies`` lie in ``[-0.5, 0.5)`` for two-sided
            or ``[0, 0.5]`` for one-sided.
        scaling: ``'density'`` (units of power / Hz) or ``'spectrum'`` (units
            of power).
        n_overlap: number of samples to overlap between segments. Defaults to
            ``len(window) // 2``.
        detrend: ``'constant'`` (subtract per-segment mean, the scipy default)
            or ``'none'``.
        return_onesided: ``True`` -> drop negative frequencies (real signals
            only). ``None`` -> auto: ``True`` for real ``x``, ``False`` for
            complex ``x``.

    Returns:
        ``(frequencies, psd)`` — ``frequencies`` of shape ``(F,)``, ``psd`` of
        shape ``x.shape[:-1] + (F,)``.
    """
    n = x.shape[-1]
    if isinstance(window, int):
        win_len = min(window, n)
        win = torch.hann_window(win_len, dtype=x.real.dtype, device=x.device)
    else:
        win = window.to(dtype=x.real.dtype, device=x.device)
        win_len = win.shape[0]
    if n_overlap is None:
        n_overlap = win_len // 2
    stride = win_len - n_overlap
    if stride <= 0:
        raise ValueError(
            f"n_overlap ({n_overlap}) must be smaller than window length ({win_len})"
        )

    if return_onesided is None:
        return_onesided = not x.is_complex()
    if return_onesided and x.is_complex():
        raise ValueError("return_onesided=True is invalid for complex signals")

    # Sliding-window view: (..., n_windows, win_len)
    segs = _window_view(x, dim=-1, size=win_len, stride=stride)

    # Optional detrending per segment
    if detrend == "constant":
        segs = segs - segs.mean(dim=-1, keepdim=True)
    elif detrend != "none":
        raise ValueError(f"detrend must be 'constant' or 'none', got {detrend!r}")

    # Apply window
    segs = segs * win

    # FFT each segment
    if return_onesided:
        spec = torch.fft.rfft(segs, n=win_len, dim=-1)
        freqs = torch.fft.rfftfreq(win_len, d=1.0 / fs).to(x.device)
    else:
        spec = torch.fft.fft(segs, n=win_len, dim=-1)
        freqs = torch.fft.fftfreq(win_len, d=1.0 / fs).to(x.device)

    # Average power across segments
    psd = (spec.real**2 + spec.imag**2).mean(dim=-2)

    # Scaling factor: matches scipy convention
    if scaling == "density":
        scale = 1.0 / (fs * (win * win).sum())
    elif scaling == "spectrum":
        scale = 1.0 / (win.sum() ** 2)
    else:
        raise ValueError(f"scaling must be 'density' or 'spectrum', got {scaling!r}")
    psd = psd * scale

    # One-sided correction: double interior bins (DC and Nyquist unchanged)
    if return_onesided:
        # Indices 1..-2 (interior of rfft output) get a factor of 2
        if psd.shape[-1] > 2:
            psd_mid = psd[..., 1:-1] * 2.0
            psd = torch.cat([psd[..., :1], psd_mid, psd[..., -1:]], dim=-1)
        elif psd.shape[-1] == 2:
            # Only DC + Nyquist
            pass

    return freqs, psd
