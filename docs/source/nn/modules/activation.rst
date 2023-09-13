Activation
==========

Complex-valued activation functions must take into account the 2 degrees-of-freedom inherent to complex-valued data, typically represented as real and imaginary parts or magnitude and phase. 
Two common generalized classes of complex-valued activation functions operate on these respective representations and are defined as *Type-A* and *Type-B* functions. 

:doc:`Type-A <./activation/split_type_A>` activation functions consist of two real-valued functions, :math:`G_{real}(\cdot)` and :math:`G_{imag}(\cdot)`, which are applied to the real and imaginary parts of the input tensor, respectively, as 

.. math::

    G(\mathbf{z}) = G_{real}(\mathbf{x}) + j G_{imag}(\mathbf{y}),

where :math:`\mathbf{z} = \mathbf{x} + j\mathbf{y}`. 

:doc:`Type-B <./activation/split_type_B>` activation functions consist of two real-valued functions, :math:`G_{mag}(\cdot)` and :math:`G_{phase}(\cdot)`, which are applied to the magnitude (modulus) and phase (angle, argument) of the input tensor, respectively, as 

.. math::

    G(\mathbf{z}) = G_{mag}(|\mathbf{z}|) * \exp(j G_{phase}(angle(\mathbf{z})))

In contrast, :doc:`fully complex activation functions <./activation/fully_complex>` fit neither the :doc:`Split Type-A <./activation/split_type_A>` or :doc:`Split Type-B <./activation/split_type_B>` designation. 

The final designation of complex-valued activation functions detailed in this work are extensions of the :doc:`Rectified Linear Unit (ReLU) to the complex plane<./activation/complex_relu>`.

.. toctree::
    :maxdepth: 2

    activation/split_type_A
    activation/split_type_B
    activation/fully_complex
    activation/complex_relu
