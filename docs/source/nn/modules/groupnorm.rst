GroupNorm
=========

Complex-valued group normalization. Splits channels into groups, applies a
2x2 whitening transform within each group, then a per-channel 2x2 affine +
2-vector bias. No running statistics (differs from :class:`BatchNorm2d`).

.. automodule:: complextorch.nn.modules.groupnorm
    :members:
