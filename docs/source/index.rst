Welcome to complextorch's documentation!
========================================

:Author:  Josiah W. Smith
:Version: |release| of 05/10/2026

A lightweight complex-valued neural network package built on PyTorch.

This is a package built on `PyTorch <https://pytorch.org/>`_ with the intention of implementing light-weight interfaces for common complex-valued neural network operations and architectures.
Notably, we include efficient implementations for linear, convolution, and attention modules in addition to activation functions and normalization layers such as batchnorm and layernorm.

Although there is an emphasis on 1-D data tensors, due to a focus on signal processing, communications, and radar data, many of the routines are implemented for 2-D and 3-D data as well.

Version 1.2 Release Notes:

* The legacy ``CVTensor`` API and its supporting helpers (``cat``, ``roll``, ``from_polar``, ``randn``, and the ``torch.Tensor.rect`` / ``torch.Tensor.polar`` monkey-patch) have been removed. The package now operates exclusively on complex-dtype ``torch.Tensor`` (typically ``torch.cfloat``). Use ``torch.polar(abs, angle)`` and ``torch.randn(..., dtype=torch.cfloat)`` directly.
* Correctness fixes in ``SlowLinear`` / ``SlowConv*`` / ``SlowConvTranspose*`` — the Gauss-trick bias was previously off by ``b_i * (1 + j)`` when ``bias=True``. ``SlowConv*`` and ``SlowConvTranspose*`` now also correctly forward ``dilation`` and ``output_padding``. The fast (native-cfloat) wrappers were unaffected.
* Complex-valued ``BatchNorm*`` eval-mode no longer broadcasts ``running_mean`` against the wrong axes.
* ``PhaseSigmoid`` is now implemented (previously was an empty class). ``MagMinMaxNorm`` now correctly preserves phase (previously it subtracted a real scalar from a complex tensor).
* Fast ``ConvTranspose1d`` / ``ConvTranspose2d`` / ``ConvTranspose3d`` are now exported from ``complextorch.nn``. Their ``output_padding`` default matches PyTorch's (``0``).
* Complex-valued losses (``CVQuadError``, ``CVFourthPowError``, ``CVCauchyError``, ``CVLogCoshError``, ``CVLogError``) now accept a ``reduction`` argument (``'mean'`` | ``'sum'`` | ``'none'``), defaulting to ``'mean'``.
* ``complextorch.nn.Conv1d`` (and its 2-D / 3-D / transposed siblings) wrap ``torch.nn.Conv1d`` with ``dtype=torch.cfloat`` for maximum efficiency. The hand-rolled real/imag-split convolutions remain available under the ``Slow`` prefix.

.. toctree::
   :maxdepth: 3
   
   installation
   nn

.. toctree::
   :maxdepth: 1

   about

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
