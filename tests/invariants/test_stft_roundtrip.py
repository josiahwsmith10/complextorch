"""Property test: InverseSTFT inverts STFT on the covered interior region.

With matching analysis/synthesis windows the window-squared overlap-add
reconstructs the original signal exactly on every sample covered by a non-zero
window tap; we compare the interior (trimming one frame from each end).
"""

from __future__ import annotations

import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from complextorch.nn.modules.frontend import STFT, InverseSTFT


@given(seed=st.integers(0, 10_000), n_extra=st.integers(4, 16))
@settings(max_examples=8, deadline=None)
def test_stft_roundtrip(seed, n_extra):
    n_fft, hop = 16, 4
    torch.manual_seed(seed)
    length = n_fft + n_extra * hop  # full coverage, out_len == length
    x = torch.randn(2, length, dtype=torch.cfloat)
    stft = STFT(n_fft=n_fft, hop_length=hop)
    istft = InverseSTFT(n_fft=n_fft, hop_length=hop)
    recon = istft(stft(x))
    torch.testing.assert_close(
        recon[..., n_fft : length - n_fft],
        x[..., n_fft : length - n_fft],
        atol=1e-4,
        rtol=1e-4,
    )
