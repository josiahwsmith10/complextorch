Initialization
==============

Variance-correct complex-valued weight initializers — drop-in analogues of
:mod:`torch.nn.init` that produce the intended :math:`\mathrm{Var}(|w|^2)`
on a complex parameter (the built-in PyTorch initializers treat the real
and imaginary parts independently, which is the wrong target for the
complex magnitude).

.. automodule:: complextorch.nn.init
    :members:
