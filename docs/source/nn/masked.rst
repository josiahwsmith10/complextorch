Masked / Pruned Layers
======================

Fixed-sparsity-pattern complex layers. Each layer carries a binary
``mask`` buffer applied to its weight at forward time. Use these for
inference-time pruning: train with :mod:`complextorch.nn.relevance`,
extract a relevance mask, then load it into a masked inference model via
:func:`complextorch.nn.masked.deploy_masks`.

Base
----

.. automodule:: complextorch.nn.masked.base
    :members:

Linear / Bilinear
-----------------

.. automodule:: complextorch.nn.masked.linear
    :members:

Conv
----

.. automodule:: complextorch.nn.masked.conv
    :members:

Module-walking helpers
----------------------

.. automodule:: complextorch.nn.masked
    :members: deploy_masks, binarize_masks, is_sparse, named_masks
