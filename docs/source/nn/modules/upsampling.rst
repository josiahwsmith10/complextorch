Upsampling
==========

Two flavors of complex-valued upsampling / interpolation:

* :class:`Upsample` — split form, interpolates real and imaginary parts independently.
* :class:`PolarUpsample` — polar form, interpolates magnitude and phase
  independently and recombines via :math:`|z| \cdot \exp(j \arg z)`.

.. automodule:: complextorch.nn.modules.upsampling
    :members:
