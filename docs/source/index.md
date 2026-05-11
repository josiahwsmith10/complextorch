---
sd_hide_title: true
---

# complextorch

```{rst-class} lead
A lightweight complex-valued neural network package built on PyTorch.
```

`complextorch` provides drop-in `complextorch.nn.*` modules whose names mirror
`torch.nn.*` ({class}`torch.nn.Conv1d`, {class}`torch.nn.Linear`, ...), so a
real-valued PyTorch model can be ported to complex-valued by changing the
import. The library emphasises 1-D signal-processing / radar / comms workloads,
but most layers are also provided for 2-D and 3-D.

::::{grid} 1 2 2 3
:gutter: 3
:margin: 4 4 0 0

:::{grid-item-card} {fas}`rocket` Getting started
:link: examples/getting_started
:link-type: doc

A runnable notebook covering the README example, activation comparisons,
and an end-to-end Conv1d demo.
:::

:::{grid-item-card} {fas}`book` Concepts
:link: concepts/activations
:link-type: doc

Type-A / Type-B / fully-complex activations, and when to reach for the
native cfloat vs. Gauss-trick (real/imag split) modules.
:::

:::{grid-item-card} {fas}`code` API reference
:link: api/complextorch/index
:link-type: doc

Auto-generated reference for every public class and function in
`complextorch`, with cross-links into PyTorch's docs.
:::

::::

## Install

```sh
pip install complextorch
```

PyTorch is **not** installed automatically — install the wheel matching your
CUDA/CPU target from <https://pytorch.org/get-started/locally/> first. See
[Installation](installation.md) for source-install and development setup.

## Why `complextorch`?

- **Native cfloat wrappers.** {class}`complextorch.nn.Conv1d`,
  {class}`complextorch.nn.Linear`, and friends are thin wrappers around
  `torch.nn` modules with `dtype=torch.cfloat`. PyTorch ≥ 2.1 has fast complex
  kernels — these are the recommended path.
- **Reference implementations on hand.** The {mod}`complextorch.nn.gauss`
  subpackage keeps the original real/imag-split Gauss-trick implementations
  ({class}`complextorch.nn.gauss.Conv1d`, etc.) around as reference math.
- **Three composition primitives.** Activations, pooling, losses, dropout, and
  softmax are built on `apply_complex`, `apply_complex_split`, and
  `apply_complex_polar` in {mod}`complextorch.nn.functional` — see
  [Activations](concepts/activations.md) for the math.
- **Beyond layers.** Includes {mod}`complextorch.signal` (a torch port of
  Welch's PSD), {mod}`complextorch.transforms` (torchcvnn-style transforms),
  {mod}`complextorch.nn.init` (variance-correct complex initializers),
  {mod}`complextorch.nn.relevance` (Variational Dropout & ARD), and
  {mod}`complextorch.nn.masked` (fixed-mask sparsified layers).

## Citation

If `complextorch` helps your research, please cite the package and consider
citing the author's [PhD thesis](https://arxiv.org/abs/2306.15341) and related
papers — see [About](about.md) for the full list.

```{toctree}
:hidden:
:caption: User guide

installation
examples/index
concepts/activations
concepts/native-vs-gauss
```

```{toctree}
:hidden:
:caption: Reference

api/complextorch/index
changelog
```

```{toctree}
:hidden:
:caption: About

about
```
