Welcome to complextorch's documentation!
========================================

:Author:  Josiah W. Smith
:Version: |release| of 05/11/2026

A lightweight complex-valued neural network package built on PyTorch.

This is a package built on `PyTorch <https://pytorch.org/>`_ with the intention of implementing light-weight interfaces for common complex-valued neural network operations and architectures.
Notably, we include efficient implementations for linear, convolution, and attention modules in addition to activation functions and normalization layers such as batchnorm and layernorm.

Although there is an emphasis on 1-D data tensors, due to a focus on signal processing, communications, and radar data, many of the routines are implemented for 2-D and 3-D data as well.

Version 2.0 Release Notes (feature-parity expansion):

* New top-level subpackages: ``complextorch.signal`` (``pwelch``), ``complextorch.transforms`` (torchcvnn-style dataloader transforms — ``LogAmplitude``, ``FFT2``, ``IFFT2``, ``FFTResize``, ``PolSAR``, ``Normalize``, ``RandomPhase``, etc.), ``complextorch.datasets`` (SAR / MRI dataset surface; ``SAMPLE`` and ``SLCDataset`` are full implementations, the SAR/MRI-specific readers are present as importable stubs with upstream pointers), and ``complextorch.models`` (Vision Transformer with ``vit_t/s/b/l/h`` presets).
* New ``complextorch.nn.init`` module: ``kaiming_normal_``, ``kaiming_uniform_``, ``xavier_normal_``, ``xavier_uniform_``, ``trabelsi_standard_``, ``trabelsi_independent_`` — variance-correct complex weight initializers (PyTorch's built-ins treat the real and imaginary parts independently, which is wrong for complex magnitude).
* New ``complextorch.nn.relevance`` (complex Variational Dropout & Automatic Relevance Determination) and ``complextorch.nn.masked`` (fixed-mask sparsified layers) subsystems for learned-sparsity workflows. Adds ``LinearVD``, ``LinearARD``, ``BilinearVD/ARD``, ``Conv{1,2,3}dVD/ARD``, plus ``LinearMasked``/``Conv*dMasked`` and the deploy/extract helpers ``named_penalties``, ``compute_ard_masks``, ``deploy_masks``. Requires ``scipy`` (new runtime dep).
* New RNN family: ``GRUCell``, ``GRU``, ``LSTMCell``, ``LSTM`` (cell-based, with optional ``batchnorm=True`` flag for stable deep stacks).
* New transformer family: ``TransformerEncoderLayer``, ``TransformerEncoder``, ``TransformerDecoderLayer``, ``TransformerDecoder``, ``Transformer``.
* New normalization: ``RMSNorm``, ``GroupNorm``, ``NaiveBatchNorm{1,2,3}d`` (split-form baseline). The functional whitening helpers (``whiten2x2_batch_norm``, ``whiten2x2_layer_norm``, ``inv_sqrtm2x2``, ``batch_norm``, ``layer_norm``) are now public in ``complextorch.nn.functional``.
* New pooling: ``MagMaxPool{1,2,3}d`` (magnitude-argmax, the canonical complex max-pool — ``torch.nn.MaxPool*d`` doesn't define ``>`` on complex), ``AvgPool{1,2,3}d``.
* New channel dropout: ``Dropout1d``, ``Dropout2d``, ``Dropout3d`` with shared real/imag mask (Trabelsi 2018).
* New upsampling: ``Upsample`` (split real/imag) and ``PolarUpsample`` (phase-preserving polar form).
* New activations: ``CELU``, ``CCELU``, ``CGELU`` (split type-A ELU/CELU/GELU + ``CVSplit*`` aliases), ``zAbsReLU``, ``zLeakyReLU`` (first-quadrant + leaky variants), ``Mod`` (magnitude as module), ``AdaptiveModReLU`` (per-channel learnable threshold). Existing ``modReLU`` gains a ``learnable=True`` flag for a scalar trainable threshold.
* New layers: ``Bilinear`` (with ``conjugate=True/False``), ``InterleavedToComplex`` / ``ComplexToInterleaved`` / ``ConcatenatedToComplex`` / ``ComplexToConcatenated`` / ``RealToComplex`` (layout-conversion modules), ``PhaseShift`` (learnable per-channel phase rotation).
* New loss: ``MSELoss`` matching ``torch.nn.MSELoss`` exactly (no 1/2 factor — distinct from ``CVQuadError``).
* **Breaking**: ``MultiheadAttention`` / ``ScaledDotProductAttention`` now use the Hermitian inner product ``QKᴴ`` (was ``QKᵀ`` — a math bug). New ``softmax_on='complex'|'real'`` flag selects the attention-weight semantics; default ``'complex'`` keeps the existing ``CVSoftMax`` behavior.
* **Breaking**: ``Linear`` / ``SlowLinear`` / fast ``Conv{1,2,3}d`` / fast ``ConvTranspose{1,2,3}d`` default ``bias=True`` to match ``torch.nn``. Pass ``bias=False`` explicitly if you relied on the old default.
* New optional dependencies (gated behind extras): ``complextorch[datasets]`` pulls in ``h5py``; ``complextorch[datasets-alos]`` pulls in ``rasterio``.

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
   signal
   transforms
   datasets
   models

.. toctree::
   :maxdepth: 1

   about

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
