<h1 align="center">ComplexTorch</h1>
<p align="center">
  <img src="https://img.shields.io/badge/complextorch-black?style=for-the-badge">
  <a href="https://github.com/josiahwsmith10/complextorch/actions/workflows/test.yml"><img src="https://github.com/josiahwsmith10/complextorch/actions/workflows/test.yml/badge.svg" alt="pytest"></a>
  <img src="https://img.shields.io/badge/coverage-100%25-brightgreen" alt="coverage 100%">
</p>
<h3>

[Homepage](https://github.com/josiahwsmith10/complextorch) | [Documentation](https://josiahwsmith10.github.io/complextorch/latest/) | [Changelog](CHANGELOG.md)

</h3>

# Complex PyTorch
(Available on [PyPI](https://pypi.org/project/complextorch/))

Author: Josiah W. Smith, Ph.D.

A lightweight complex-valued neural network package built on PyTorch.

This is a package built on [PyTorch](https://pytorch.org/) with the intention of implementing light-weight interfaces for common complex-valued neural network operations and architectures. 
Notably, we include efficient implementations for linear, convolution, and attention modules in addition to activation functions and normalization layers such as batchnorm and layernorm.

Although there is an emphasis on 1-D data tensors, due to a focus on signal processing, communications, and radar data, many of the routines are implemented for 2-D and 3-D data as well.

## What's new

See [CHANGELOG.md](CHANGELOG.md) for the full release history.

**Version 2.2.0** is a correctness release fixing all 28 defects confirmed by
a full-repository code review — several were silently-wrong numerics
(`RMSNorm` halved output power at init, `trabelsi_independent_` produced
vanishing weights, `pwelch` halved the top PSD bin for odd windows, the
Gauss-trick convolutions ignored `padding_mode` in their cross term). It also adds
**full `torch.nn.Transformer`-parity attention masking** (`attn_mask`,
`key_padding_mask`, `src/tgt/memory` masks, and
`Transformer.generate_square_subsequent_mask` for causal decoding, with
well-defined semantics under the complex softmax variants). Three changes are
breaking: the transformer stack now defaults to `batch_first=False` (torch
parity), `ViTLayer` is now genuinely pre-norm (old ViT checkpoints do not
load), and `CVSigmoid` was removed (use `torch.nn.Sigmoid`, which supports
complex natively).

Version 2.1 added modern complex-valued architectures (state-space models,
unitary RNNs, holographic attention, KAN, time-frequency frontends). Version
2.0 brought major feature-parity expansion (RNN/LSTM, Transformer,
ARD/Variational Dropout, masked layers, transforms, signal, datasets, models
subpackages); see the changelog for the breaking changes around
`MultiheadAttention` and default `bias=True`. Version 2.0.1 added
**symmetry-aware modules** (U(1)-equivariant / invariant building blocks from
[SurReal](https://arxiv.org/abs/1910.11334) and
[CDS](https://arxiv.org/abs/2112.01525), with reference networks under
`complextorch.models`) plus `SpectralPool{1,2,3}d`
([Rippel 2015](https://arxiv.org/abs/1506.03767) /
[Trabelsi 2018](https://arxiv.org/abs/1705.09792)) for frequency-domain
downsampling.

## Documentation

Live docs: <https://josiahwsmith10.github.io/complextorch/latest/> — including
an executable [Getting Started notebook](https://josiahwsmith10.github.io/complextorch/latest/examples/getting_started.html)
and a full API reference auto-generated from docstrings. The accompanying
paper is at `docs/complextorch_paper.pdf`.

## Dependencies

This library requires numpy and PyTorch.[PyTorch](https://pytorch.org/get-started/locally/) should be installed to your environment using the compute platform (CPU/GPU) settings for your machine. PyTorch will not be automatically installed with the installation of complextorch and MUST be installed manually by the user.

## Installation:

IMPORTANT:
Prior to installation, [install PyTorch](https://pytorch.org/get-started/locally/) to your environment using your preferred method using the compute platform (CPU/GPU) settings for your machine.

Using [pip](https://pypi.org/project/complextorch/)

```sh
pip install complextorch
```
From the source:
```sh
git clone https://github.com/josiahwsmith10/complextorch.git
cd complextorch
pip install -r requirements.txt
pip install . --use-pep517
```

## Basic Usage

``` python
import torch
import complextorch as cT

x = torch.randn(64, 5, 7, dtype=torch.cfloat)
model = cT.nn.Conv1d(5, 16, kernel_size=3)
y = model(x)
```

## Development

The test suite mirrors `complextorch/` 1:1 under `tests/` and covers every public class and helper. CI enforces **100% line coverage** on Python 3.10 / 3.11 / 3.12 — any PR that drops coverage fails automatically.

```sh
pip install '.[test]'                                   # pytest, pytest-cov, pytest-xdist, hypothesis
pytest                                                  # auto-parallel (-n auto) from pyproject
pytest --cov=complextorch --cov-report=term-missing --cov-fail-under=100   # mirror CI exactly
pytest --cov=complextorch --cov-report=html && open htmlcov/index.html     # browse uncovered lines
```

When adding a new module, add a matching `tests/.../test_<module>.py`. Fast/Slow numerical equivalence checks share weights via `load_state_dict`; loss tests sweep the `reduction` matrix; round-trip invariants (Fast/Slow, polar, casting, FFT) live under `tests/invariants/` and use Hypothesis. Prefer per-line `# pragma: no cover` over whole-function exclusions so dead code stays visible.
