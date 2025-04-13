Split Type-A Activation Functions
=================================

*Type-A* activation functions consist of two real-valued functions, :math:`G_\mathbb{R}(\cdot)` and :math:`G_\mathbb{I}(\cdot)`, which are applied to the real and imaginary parts of the input tensor, respectively, as 

.. math::

    G(\mathbf{z}) = G_\mathbb{R}(\mathbf{x}) + j G_\mathbb{I}(\mathbf{y}),

where :math:`\mathbf{z} = \mathbf{x} + j\mathbf{y}`. 

In most cases, :math:`G_\mathbb{R}(\cdot) = G_\mathbb{I}(\cdot)`; however, :math:`G_\mathbb{R}(\cdot)` and :math:`G_\mathbb{I}(\cdot)` can also be distinct functions. 
A generalized Type-A split activation function is defined in :class:`GeneralizedSplitActivation`, which accepts two real-valued torch.nn.Module objects for :math:`G_\mathbb{R}(\cdot)` and :math:`G_\mathbb{I}(\cdot)`, respectively. 

.. automodule:: complextorch.nn.modules.activation.split_type_A
    :members:
