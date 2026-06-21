# Learnable time-frequency front-ends

These modules turn a raw 1-D signal (real **or** complex) into a native complex
time-frequency representation, with learnable parameters so the front-end trains
end-to-end with the rest of the model. They complement
{func}`complextorch.signal.pwelch` and the {class}`complextorch.nn.FFTBlock`
family.

## Learnable STFT

{class}`complextorch.nn.STFT` frames the signal, applies a **learnable window**
(initialised to Hann), and FFTs each frame, returning a complex spectrogram of
shape `(..., n_fft, n_frames)`. {class}`complextorch.nn.InverseSTFT` inverts it
with a window-squared overlap-add, so with matching windows the round-trip is
exact on every sample covered by a non-zero window tap.

```python
import torch
import complextorch as ctorch

stft  = ctorch.nn.STFT(n_fft=64, hop_length=16)
istft = ctorch.nn.InverseSTFT(n_fft=64, hop_length=16)
istft.window = stft.window   # tie the windows so the inverse stays exact once trained

x = torch.randn(2, 1024, dtype=torch.cfloat)   # complex baseband signal
spec = stft(x)                                  # (2, 64, n_frames), complex
recon = istft(spec)
print("spectrogram:", spec.shape, spec.dtype)
print("reconstruction error (interior):",
      (recon[..., 64:-64] - x[..., 64:-64]).abs().max().item())
```

## Learnable complex filterbanks (Gabor / Morlet)

{class}`complextorch.nn.ComplexGaborConv1d` is a complex, wavelet-style analogue
of SincNet: each output filter is a windowed complex exponential

$$
g_o(t) = e^{-t^2/(2\sigma_o^2)}\, e^{j 2\pi f_o t}
$$

with a **learnable** centre frequency $f_o$ and bandwidth $\sigma_o$, applied
with a complex 1-D convolution. {class}`complextorch.nn.MorletConv1d` is the
zero-mean (admissible) variant — it subtracts the envelope-weighted mean so the
filter has no DC response.

```python
gabor = ctorch.nn.ComplexGaborConv1d(in_channels=1, out_channels=32,
                                     kernel_size=63, padding=31)
y = gabor(x.unsqueeze(1))   # (2, 32, 1024), complex
print(y.shape, y.dtype)
```
