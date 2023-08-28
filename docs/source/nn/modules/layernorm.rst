Layer Normalization
===================

Similar to :doc:`batch normalization <./batchnorm>`, layer normalization is a crucial element to the convergence and robustness of many deep learning applications; however, its implementation must be carefully address for complex-valued data. 
The complex-valued corollary to zero-mean unit variance normalization is known as whitening.

Additional details can be found in the following paper:

    **J. A. Barrachina, C. Ren, G. Vieillard, C. Morisseau, and J.-P. Ovarlez. Theory and Implementation of Complex-Valued Neural Networks.**

        - Section 6

        - https://arxiv.org/abs/2302.08286

.. automodule:: complextorch.nn.modules.layernorm
    :members:
