---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Getting started

This notebook is **executed on every docs build** — if it stops running
against the latest `complextorch`, CI fails. Treat it as a smoke-test of the
public API as well as a tutorial.

## 1 · Imports & version check

```{code-cell}
import torch
import complextorch as ctorch

print(f"torch        {torch.__version__}")
print(f"complextorch {ctorch.__version__}")
```

## 2 · Building a complex tensor

`complextorch` operates on complex-dtype `torch.Tensor` (typically
`torch.cfloat`). There is no special wrapper type — use PyTorch's built-ins
directly:

```{code-cell}
torch.manual_seed(0)

x = torch.randn(8, 5, 16, dtype=torch.cfloat)   # (batch, channels, length)
print(x.shape, x.dtype)
print(x[0, 0, :3])
```

You can construct from magnitude / phase via `torch.polar`:

```{code-cell}
mag   = torch.rand(8, 5, 16)
phase = torch.rand(8, 5, 16) * (2 * torch.pi) - torch.pi
z = torch.polar(mag, phase)
print(z.dtype, z[0, 0, 0])
```

## 3 · Conv1d + Linear (the README example)

The native cfloat modules (`Conv1d`, `Linear`, ...) are thin wrappers around
`torch.nn` with `dtype=torch.cfloat`. See
[Native vs. Gauss-trick modules](../concepts/native-vs-gauss.md) for the design rationale.

```{code-cell}
conv = ctorch.nn.Conv1d(in_channels=5, out_channels=16, kernel_size=3)
fc   = ctorch.nn.Linear(in_features=16 * 14, out_features=4)

h = conv(x)                       # (8, 16, 14)
h_flat = h.reshape(h.size(0), -1) # (8, 16*14)
y = fc(h_flat)                    # (8, 4)

print("conv output:", h.shape, h.dtype)
print("fc output:  ", y.shape, y.dtype)
```

Both modules accept and emit complex tensors — and gradients flow through
them just like any real-valued `torch.nn` module:

```{code-cell}
loss = y.abs().pow(2).mean()
loss.backward()

total_grad_norm = sum(p.grad.abs().pow(2).sum() for p in conv.parameters()).sqrt()
print(f"loss = {loss.item():.4f}, conv grad norm = {total_grad_norm:.4f}")
```

## 4 · Type-A vs. Type-B activations

The package implements two paradigms for complex activations (see
[Activations](../concepts/activations.md) for the math). Let's compare a
Type-A `CVSplitReLU` (independent real/imag) against a Type-B `modReLU`
(magnitude-only) on the same input.

```{code-cell}
import matplotlib.pyplot as plt

z = torch.complex(
    real=torch.linspace(-2, 2, 200).repeat(200, 1),
    imag=torch.linspace(-2, 2, 200).repeat(200, 1).T,
)

split_relu = ctorch.nn.CVSplitReLU()
mod_relu   = ctorch.nn.modReLU(bias=-0.5)

with torch.no_grad():
    a = split_relu(z)
    b = mod_relu(z)

fig, axes = plt.subplots(2, 2, figsize=(8, 7), sharex=True, sharey=True)
for ax, data, title in zip(
    axes.flat,
    [a.abs(), a.angle(), b.abs(), b.angle()],
    ["CVSplitReLU |·|", "CVSplitReLU ∠", "modReLU |·|", "modReLU ∠"],
):
    im = ax.imshow(data, extent=[-2, 2, -2, 2], origin="lower",
                   cmap="twilight" if "∠" in title else "viridis")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
axes[1, 0].set_xlabel("Re(z)"); axes[1, 1].set_xlabel("Re(z)")
axes[0, 0].set_ylabel("Im(z)"); axes[1, 0].set_ylabel("Im(z)")
plt.tight_layout();
```

`CVSplitReLU` zeros the real/imag components independently — it doesn't
preserve phase. `modReLU` only modulates magnitude (`|z| - b`)+ and leaves the
phase untouched.

## 5 · Welch's PSD on a complex signal

{func}`complextorch.signal.pwelch` is a torch port of `scipy.signal.welch`
that's differentiable end-to-end — so it can sit inside a loss function.

```{code-cell}
from complextorch.signal import pwelch

t = torch.linspace(0, 1, 4096)
sig = torch.exp(1j * 2 * torch.pi * 50 * t).to(torch.cfloat) \
    + 0.5 * torch.exp(1j * 2 * torch.pi * 120 * t).to(torch.cfloat) \
    + 0.1 * torch.randn(4096, dtype=torch.cfloat)

f, psd = pwelch(sig, fs=4096.0, window=256, n_overlap=128)

plt.figure(figsize=(7, 3))
plt.semilogy(f.numpy(), psd.numpy())
plt.xlabel("Frequency (Hz)"); plt.ylabel("PSD"); plt.title("pwelch demo")
plt.tight_layout();
```

The two tones at 50 Hz and 120 Hz should be clearly visible. Because `pwelch`
is autograd-friendly, you can use the PSD as a spectral loss for training a
complex-valued generator network.

## 6 · Spectral pooling

{class}`complextorch.nn.SpectralPool2d` (and its 1-D / 3-D siblings)
downsamples by truncating the centered discrete Fourier spectrum — a
complex-valued port of the spectral pooling layer from Rippel et al. (2015)
and Trabelsi et al. (2018). It preserves the DC bin exactly, so the
spatial mean is unchanged.

```{code-cell}
import torch
import complextorch as ctorch

torch.manual_seed(0)
x = torch.randn(2, 3, 16, 16, dtype=torch.cfloat)
pool = ctorch.nn.SpectralPool2d((8, 8))
y = pool(x)

# Mean preservation: spectral pooling routes DC through unchanged.
mean_err = (y.mean(dim=(-2, -1)) - x.mean(dim=(-2, -1))).abs().max().item()
print(f"input  shape {tuple(x.shape)}")
print(f"output shape {tuple(y.shape)}")
print(f"max |mean(y) - mean(x)| = {mean_err:.2e}")
```

Because the operator is a linear function of the input (an FFT, a centered
crop, and an IFFT), gradients flow back through it like any other layer:

```{code-cell}
x = torch.randn(2, 3, 16, 16, dtype=torch.cfloat, requires_grad=True)
y = ctorch.nn.SpectralPool2d((8, 8))(x)
y.abs().pow(2).sum().backward()
print(f"x.grad shape {tuple(x.grad.shape)}, all finite = {torch.isfinite(x.grad).all().item()}")
```

## Where next?

- Browse the [API reference](../api/complextorch/index) for the full module
  surface (`nn`, `signal`, `transforms`, `datasets`, `models`).
- Read the [Activations](../concepts/activations.md) deep-dive for Type-A /
  Type-B / fully-complex / ReLU-variant theory.
- Check the [changelog](../changelog.md) for what landed in the current
  release.
