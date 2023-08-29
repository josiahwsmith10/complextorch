Manifold-Based Layers
=====================

In a paper title `Complex-Valued Learning as Principled Transformations on a Scaling and Rotation Manifold`, the authors R Chakraborty, Y Xing, and S Yu introduce a complex-valued convolution operator offering similar equivariance properties to the spatial equivariance of the traditional real-valued convolution operator.
By approach the complex domain as a Riemannian homogeneous space consisting of the product of planar rotation and non-zero scaling, they define a convolution operator equivariant to phase shift and amplitude scaling.
Although their paper shows promising results in reducing the number of parameters of a complex-valued network for several problems, their work has not gained mainstream support.

As the authors mention in the final bullet point in Section IV-A1,

    If :math:`d` is the manifold distance in (2) for the Euclidean
    space that is also Riemannian, then wFM has exactly the
    weighted average as its closed-form solution. That is, our
    wFM convolution on the Euclidean manifold is reduced
    to the standard convolution, although with the additional
    convexity constraint on the weights.

Hence, the implementation closely follows the conventional convolution operator with the exception of the weight normalization.

Note: the weight normalization, although consistent with the authors' implementation, lacks adequate explanation from the literature and could be improved for further clarity.

Based on work from the following paper:

    **R Chakraborty, Y Xing, S Yu. SurReal: Complex-Valued Learning as Principled Transformations on a Scaling and Rotation Manifold**

        - Eqs. (14)-(16)

        - https://arxiv.org/abs/1910.11334

        - Modified from implementation: https://github.com/xingyifei2016/RotLieNet (yields consistent results as this implementation)

.. automodule:: complextorch.nn.modules.manifold
    :members:
