Split Type-B Activation Functions
=================================

*Type-B* activation functions consist of two real-valued functions, :math:`G_{mag}(\cdot)` and :math:`G_{phase}(\cdot)`, which are applied to the magnitude (modulus) and phase (angle, argument) of the input tensor, respectively, as 

.. math::

    G(\mathbf{z}) = G_{mag}(|\mathbf{z}|) * \exp(j G_{phase}(\\text{angle}(\mathbf{z})))

A generalized Type-B split activation function is defined in :class:`GeneralizedPolarActivation`, which accepts two real-valued torch.nn.Module objects for :math:`G_{mag}(\cdot)` and :math:`G_{phase}(\cdot)`, respectively. 


.. automodule:: complextorch.nn.modules.activation.split_type_B
    :members:
