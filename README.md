<h1 align="center">ComplexTorch</h1>
<p align="center">
  <img src="https://img.shields.io/badge/complextorch-black?style=for-the-badge">
  [![GitHub Repo stars](https://img.shields.io/github/stars/tinygrad/tinygrad)](https://github.com/tinygrad/tinygrad/stargazers)
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

## Documentation

Please see [Read the Docs](https://complextorch.readthedocs.io/en/latest/index.html) or our [arXiv](https://github.com/josiahwsmith10/complextorch) paper, which is also located at ```docs/complextorch_paper.pdf```.

## Dependencies

This library requires numpy and PyTorch.[PyTorch](https://pytorch.org/get-started/locally/) should be installed to your environment using the compute platform (CPU/GPU) settings for your machine. PyTorch will not be automatically installed with the installation of complextorch and MUST be installed manually by the user.

## Installation:

IMPORTANT:
Prior to installation, [install PyTorch](https://pytorch.org/get-started/locally/) to your environment using your preferred method using the compute platform (CPU/GPU) settings for your machine.

Using [pip](https://pypi.org/project/complextorch/)

```
pip install complextorch
```
From the source:
```
git clone https://github.com/josiahwsmith10/complextorch.git
cd complextorch
pip install -r requirements.txt
pip install . --use-pep517
```

## Basic Usage

``` python
import complextorch as cvtorch

x = cvtorch.randn(64, 5, 7)
```
