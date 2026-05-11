<h1 align="center">ComplexTorch</h1>
<p align="center">
  <img src="https://img.shields.io/badge/complextorch-black?style=for-the-badge">
  
</p>
<h3>

[Homepage](https://github.com/josiahwsmith10/complextorch) | [Documentation](https://complextorch.readthedocs.io/en/latest/)

</h3>

# Complex PyTorch
(Available on [PyPI](https://pypi.org/project/complextorch/))

Author: Josiah W. Smith, Ph.D.

A lightweight complex-valued neural network package built on PyTorch.

This is a package built on [PyTorch](https://pytorch.org/) with the intention of implementing light-weight interfaces for common complex-valued neural network operations and architectures. 
Notably, we include efficient implementations for linear, convolution, and attention modules in addition to activation functions and normalization layers such as batchnorm and layernorm.

Although there is an emphasis on 1-D data tensors, due to a focus on signal processing, communications, and radar data, many of the routines are implemented for 2-D and 3-D data as well.

### Version 1.2 Release Notes:
- The legacy `CVTensor` API and its supporting helpers (`cat`, `roll`, `from_polar`, `randn`, and the `torch.Tensor.rect` / `torch.Tensor.polar` monkey-patch) have been removed. The package now operates exclusively on complex-dtype `torch.Tensor` (typically `torch.cfloat`). Use `torch.polar(abs, angle)` and `torch.randn(..., dtype=torch.cfloat)` directly.
- Correctness fixes in `SlowLinear` / `SlowConv*` / `SlowConvTranspose*` — the Gauss-trick bias was previously off by `b_i * (1 + j)` when `bias=True`. `SlowConv*` and `SlowConvTranspose*` now correctly forward `dilation` and `output_padding` too. The fast (native-cfloat) wrappers were unaffected.
- Complex-valued `BatchNorm*` eval-mode no longer broadcasts `running_mean` against the wrong axes.
- `PhaseSigmoid` is now implemented (previously was an empty class). `MagMinMaxNorm` now correctly preserves phase.
- Fast `ConvTranspose1d` / `ConvTranspose2d` / `ConvTranspose3d` are now exported from `complextorch.nn`. Their `output_padding` default matches PyTorch's (`0`).
- Complex-valued losses (`CVQuadError`, `CVFourthPowError`, `CVCauchyError`, `CVLogCoshError`, `CVLogError`) now accept a `reduction` argument (`'mean'` | `'sum'` | `'none'`), defaulting to `'mean'`.
- `complextorch.nn.Conv1d` (and its 2-D / 3-D / transposed siblings) wrap `torch.nn.Conv1d` with `dtype=torch.cfloat` for maximum efficiency. The hand-rolled real/imag-split convolutions remain available under the `Slow` prefix.

## Documentation

Please see [Read the Docs](https://complextorch.readthedocs.io/en/latest/index.html) or our [arXiv](https://github.com/josiahwsmith10/complextorch) paper, which is also located at ```docs/complextorch_paper.pdf```.

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
