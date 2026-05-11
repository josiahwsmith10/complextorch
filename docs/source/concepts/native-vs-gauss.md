# Native vs. Gauss-trick modules

Convolution and linear layers in `complextorch.nn` exist in two variants:

| Native cfloat (recommended) | Gauss-trick (reference) |
| --- | --- |
| {class}`complextorch.nn.Linear` | {class}`complextorch.nn.gauss.Linear` |
| {class}`complextorch.nn.Conv1d` / `Conv2d` / `Conv3d` | `complextorch.nn.gauss.Conv1d` / `Conv2d` / `Conv3d` |
| `ConvTranspose1d` / `2d` / `3d` | `complextorch.nn.gauss.ConvTranspose1d` / `2d` / `3d` |

```{note}
Up to `complextorch < 2.0` the Gauss-trick variants lived at the top level as
`SlowConv*` / `SlowLinear`. The prefix was a misleading legacy from when they
were *faster* than the naive split; they have since been moved to the
{mod}`complextorch.nn.gauss` subpackage and the `Slow` names removed.
```

## What's the difference?

**Native cfloat modules** are thin wrappers around the corresponding `torch.nn`
module constructed with `dtype=torch.cfloat`. They rely on PyTorch's native
complex kernels (available since PyTorch 2.1) and are the recommended path for
all new code.

```python
import torch
import complextorch as ctorch

x = torch.randn(8, 5, 7, dtype=torch.cfloat)
y = ctorch.nn.Conv1d(5, 16, kernel_size=3)(x)   # native cfloat kernel
```

**Gauss-trick modules** are the original hand-rolled implementations that
split each complex tensor into real and imaginary parts and apply Gauss'
multiplication trick:

$$
(R + jI)(x + jy) = (Rx - Iy) + j(Ry + Ix)
$$

with a three-multiply real-valued formulation under the hood. They predate
PyTorch's native complex support and are kept for two reasons:

1. **Reference math** — the Gauss path is the easiest place to read the
   real/imag split when you're learning the package internals or implementing
   a new layer.
2. **Explicit split parameters** — `conv_r` / `conv_i` (or `linear_r` /
   `linear_i`) are exposed as separate `nn.Module` children, which is useful
   if you want to apply different parameterizations or constraints to each
   half.

## Which should I use?

Use the **native cfloat** variant. The Gauss-trick path no longer offers a
speed advantage since PyTorch 2.1, so its only remaining role is as a
numerically-equivalent reference. The test suite under `tests/invariants/`
checks the two paths agree to floating-point tolerance on the same weights.

If you're adding a new layer that has a native PyTorch complex equivalent,
follow the native pattern (wrap `torch.nn.X` with `dtype=torch.cfloat`)
rather than reimplementing the real/imag split.

## The three composition primitives

Most non-convolutional layers in `complextorch` are built on three helpers in
{mod}`complextorch.nn.functional`:

- {func}`~complextorch.nn.functional.apply_complex` — the "naive" complex
  linear lift: $(R(x_r) - I(x_i)) + j(R(x_i) + I(x_r))$.
- {func}`~complextorch.nn.functional.apply_complex_split` — **Type-A** split:
  apply two separate functions to real and imaginary parts independently. Used
  by `CVSplit*` activations, `Dropout`, `CVSoftMax`, `AdaptiveAvgPool*d`.
- {func}`~complextorch.nn.functional.apply_complex_polar` — **Type-B** polar
  split: apply functions to magnitude and phase separately, recombine via
  `torch.polar`. Used by `CVPolar*` / `modReLU` activations.

See [Activations](activations.md) for the math behind Type-A / Type-B.

```{tip}
Construct magnitude/phase tensors with `torch.polar(abs, angle)` — it's been
a PyTorch builtin since 1.8 and is the idiomatic call. `complextorch` does
not provide a `from_polar` helper.
```
