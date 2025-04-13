Welcome to complextorch's documentation!
========================================

:Author:  Josiah W. Smith
:Version: 1.1.0 of 04/13/2025

A lightweight complex-valued neural network package built on PyTorch. 

This is a package built on `PyTorch <https://pytorch.org/>`_ with the intention of implementing light-weight interfaces for common complex-valued neural network operations and architectures. 
Notably, we include efficient implementations for linear, convolution, and attention modules in addition to activation functions and normalization layers such as batchnorm and layernorm.

Although there is an emphasis on 1-D data tensors, due to a focus on signal processing, communications, and radar data, many of the routines are implemented for 2-D and 3-D data as well.

Version 1.1.0 Release Notes:
* Methods have been renamed to reflect identical names in PyTorch, e.g., `complextorch.nn.CVConv1d` was renamed to `complextorch.nn.Conv1d`. This change was implemented for quick conversion from PyTorch to `complextorch`. 
* Use of `torch.Tensor` is now recommended over `complextorch.CVTensor`. Previous speed advantages of `complextorch.CVTensor` are no longer present if using a version of PyTorch newer than 2.1.0. 
* Similarly, previous implementations of `complextorch.nn.Conv1d` (for 1-D, 2-D, 3-D, and transposed convolution) and `complextorch.nn.Lienar` have been renamed with the prefix `Slow` as PyTorch's native convolution and linear operators now outperform that of `complextorch`. Now, `complextorch.nn.Conv1d`, for example, uses `torch.nn.Conv1d` with `dtype=torch.float` for maximum efficiency. 

.. toctree::
   :maxdepth: 3
   
   installation
   nn

.. toctree::
   :maxdepth: 1

   _cvtensor
   _cvrandom
   about

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
