# Complex PyTorch
Author: Josiah W. Smith, Ph.D.

A lightweight complex-valued neural network package built on PyTorch.

This is a package built on [PyTorch](https://pytorch.org/) with the intention of implementing light-weight interfaces for common complex-valued neural network operations and architectures. 
Notably, we include efficient implementations for linear, convolution, and attention modules in addition to activation functions and normalization layers such as batchnorm and layernorm.

Although there is an emphasis on 1-D data tensors, due to a focus on signal processing, communications, and radar data, many of the routines are implemented for 2-D and 3-D data as well.

## Documentation

Please see [Read the Docs](https://complextorch.readthedocs.io/en/latest/index.html)

## Dependencies

This library requires numpy and PyTorch.[PyTorch](https://pytorch.org/get-started/locally/) should be installed to your environment using the compute platform (CPU/GPU) settings for your machine. PyTorch will not be automatically installed with the installation of complextorch and MUST be installed manually by the user.

## Instalation:

IMPORTANT:
Prior to installation, [install PyTorch](https://pytorch.org/get-started/locally/) to your environment using your preferred method using the compute platform (CPU/GPU) settings for your machine.

Using [pip](https://pypi.org/project/complextorch/)

```
pip install complextorch
```

## Basic Usage

``` python
import complextorch as cvtorch

x = cvtorch.randn(64, 5, 7)
```