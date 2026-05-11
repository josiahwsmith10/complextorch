# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this package is

`complextorch` is a lightweight complex-valued neural network library built on PyTorch (PyPI: `complextorch`). It provides drop-in `complextorch.nn.*` modules whose names mirror `torch.nn.*` (e.g. `complextorch.nn.Conv1d`, `complextorch.nn.Linear`) so a real-valued PyTorch model can be ported by changing the import. There is an emphasis on 1-D tensors (signal processing / radar / comms) but most layers are also provided for 2-D and 3-D.

## Common commands

PyTorch is **not** declared as a normal install dep in practice — `requirements.txt` pins `torch>=1.11.0+cu115` (a CUDA build), so installing it via `pip install -r requirements.txt` will usually fail on non-CUDA machines. Install torch separately first (see https://pytorch.org/get-started/locally/), then:

```sh
pip install . --use-pep517           # install the package from source
```

Docs (Sphinx, hosted on Read the Docs, configured by `.readthedocs.yaml`):

```sh
pip install -r docs/requirements.txt
cd docs && make html                  # output in docs/build/html
```

There is **no test suite, linter config, or CI** in the repo. Don't fabricate test/lint commands — if a change needs verification, write an ad-hoc script or ask the user.

## Releasing a new version

`complextorch/__init__.py:__version__` is the single source of truth. `setup.py` parses it via regex, `docs/source/conf.py` does the same, and `docs/source/index.rst` displays it via the Sphinx `|release|` substitution. To bump the version, edit `__init__.py` only.

The release-date string in `docs/source/index.rst` (`:Version: |release| of <date>`) and the release-notes section there are still per-release manual edits.

## Architecture

### Tensor convention

Every module and helper in the package operates on complex-dtype `torch.Tensor` (typically `torch.cfloat`). The earlier `CVTensor` class (real/imag stored as two separate tensors) was removed when its speed advantages disappeared with PyTorch 2.1.0's native complex kernels. If you see `CVTensor` referenced in old release notes, that's historical — the type no longer exists in the public API.

### Fast vs. Slow modules

Many module names appear in two variants:

- `Conv1d`, `Conv2d`, `Conv3d`, `Linear` — thin wrappers around `torch.nn.Conv*` / `torch.nn.Linear` with `dtype=torch.cfloat`. These rely on PyTorch's native complex kernels and are the recommended path.
- `SlowConv1d`, …, `SlowConvTranspose3d`, `SlowLinear` — the original hand-rolled real/imag-split implementations using Gauss' trick. Kept for backwards compatibility and as reference implementations; do not delete or "modernize" them without checking with the user.

When adding a new layer that has a native PyTorch complex equivalent, follow the `Conv1d` pattern (wrap `torch.nn.X` with `dtype=torch.cfloat`) rather than reimplementing the real/imag split.

### Three composition primitives in `nn/functional.py`

Most activation, pooling, loss, dropout, and softmax modules are thin wrappers around one of three helpers, which is the main pattern to recognize when reading or extending this codebase:

- `apply_complex(real_module, imag_module, x)` — the "naive" complex linear lift: `(R(x.real) - I(x.imag)) + j(R(x.imag) + I(x.real))`.
- `apply_complex_split(r_fun, i_fun, x)` — **Type-A** split: apply two separate functions to real and imaginary parts independently. Used by `CVSplit*` activations, `Dropout`, `CVSoftMax`, `AdaptiveAvgPool*d`.
- `apply_complex_polar(mag_fun, phase_fun, x)` — **Type-B** polar split: apply functions to magnitude and phase separately, recombine via `torch.polar`. Used by `CVPolar*` / `modReLU` activations. Passing `phase_fun=None` is an optimization that skips the polar round-trip.

Magnitude/phase tensor construction goes through `torch.polar(abs, angle)` (PyTorch built-in since 1.8). Don't reintroduce a `from_polar` helper — `torch.polar` is the idiomatic call.

### Subpackage layout

- `complextorch/__init__.py` — package surface; the only thing exposed is the `nn` subpackage. There are no top-level helper functions; users should call `torch.randn(..., dtype=torch.cfloat)` and `torch.polar(...)` directly.
- `complextorch/nn/functional.py` — the three `apply_complex*` primitives plus `inv_sqrtm2x2`, `batch_norm`, `layer_norm` helpers used by the normalization layers.
- `complextorch/nn/modules/` — one file per concept (`conv.py`, `linear.py`, `batchnorm.py`, `layernorm.py`, `dropout.py`, `loss.py`, `pooling.py`, `softmax.py`, `mask.py`, `manifold.py`, `fft.py`) plus subpackages `activation/` (split Type-A, split Type-B, fully-complex, complex ReLU variants) and `attention/` (`MultiheadAttention`, `ScaledDotProductAttention`, ECA, MCA variants).
- `complextorch/nn/__init__.py` is the public surface; every user-visible class is re-exported there. When adding a new module, also add it to this file.

### Docs

Sphinx sources are in `docs/source/`, one `.rst` per module mirroring the package layout (`nn.rst`, `nn/*.rst`). New public classes should get a corresponding autodoc entry.
