Complex-Valued Rectified Linear Units
=====================================

The *Rectified Linear Unit (ReLU)* is the most common activation function in modern data-driven algorithms. 
Hence, it is oft-extended to the complex domain. 
However, whereas its nonlinearity lends itself naturally to the real domain, its application to the complex domain, specifically its activation in different quadrants of the complex plane, has led to further investigation. 

These variants of the complex-valued ReLU are all *Type-A* split activation functions, meaning they apply a function separately to the real and imaginary parts of the input tensor, as detailed in :doc:`Split Type-A <./split_type_A>`.

.. automodule:: complextorch.nn.modules.activation.complex_relu
    :members:
