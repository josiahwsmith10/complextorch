Welcome to complextorch's documentation!
========================================

:Author:  Josiah W. Smith
:Version: 1.0.0 of 08/30/2023

A lightweight complex-valued neural network package built on PyTorch. 

This is a package built on `PyTorch <https://pytorch.org/>`_ with the intention of implementing light-weight interfaces for common complex-valued neural network operations and architectures. 
Notably, we include efficient implementations for linear, convolution, and attention modules in addition to activation functions and normalization layers such as batchnorm and layernorm.

Although there is an emphasis on 1-D data tensors, due to a focus on signal processing, communications, and radar data, many of the routines are implemented for 2-D and 3-D data as well.

.. toctree::
   :maxdepth: 3
   
   installation
   nn
   _cvtensor
   _cvrandom
   about

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
