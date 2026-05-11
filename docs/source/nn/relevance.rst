Variational Dropout / ARD (Relevance)
=====================================

Complex-valued layers that learn a per-weight relevance score during
training via the local reparameterization trick. Each layer exposes:

* :attr:`BaseARD.penalty` — differentiable KL divergence to add to the loss
* :meth:`BaseARD.relevance` — post-training binary keep/drop mask

Combined with :mod:`complextorch.nn.masked`, this gives a full workflow:
train with VD/ARD, extract a relevance mask, then deploy it onto a masked
inference model.

Adapted from :mod:`cplxmodule.nn.relevance` for native ``torch.cfloat``.
Adds ``scipy`` as a runtime dependency (for the exact ``Ei``-based KL).

Base + module-walking helpers
-----------------------------

.. automodule:: complextorch.nn.relevance.base
    :members:

Linear / Bilinear VD & ARD
--------------------------

.. automodule:: complextorch.nn.relevance.linear
    :members:

Conv VD & ARD
-------------

.. automodule:: complextorch.nn.relevance.conv
    :members:
