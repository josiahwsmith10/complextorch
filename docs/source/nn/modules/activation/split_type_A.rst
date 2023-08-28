Split Type-A Activation Functions
=================================

*Type-A* activation functions consist of two real-valued functions, :math:`G_{real}(\cdot)` and :math:`G_{imag}(\cdot)`, which are applied to the real and imaginary parts of the input tensor, respectively, as 

.. math::

    G(\mathbf{z}) = G_{real}(\mathbf{z}_{real}) + j G_{imag}(\mathbf{z}_{imag})

In most cases, :math:`G_{real}(\cdot) = G_{imag}(\cdot)`; however, :math:`G_{real}(\cdot)` and :math:`G_{imag}(\cdot)` can also be distinct functions. 
A generalized Type-A split activation function is defined in :class:`GeneralizedSplitActivation`, which accepts two real-valued torch.nn.Module objects for :math:`G_{real}(\cdot)` and :math:`G_{imag}(\cdot)`, respectively. 

.. automodule:: complextorch.nn.modules.activation.split_type_A
    :members:
