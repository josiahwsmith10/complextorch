# Complex-Valued KANs

Kolmogorov-Arnold Networks (KANs) replace the fixed scalar weights of an MLP
with **learnable univariate functions** on the edges.
{class}`complextorch.nn.CVKANLayer` is a complex adaptation that places a
learnable radial-basis expansion over the **complex plane**, and
{class}`complextorch.models.CVKAN` stacks those layers into a network.

## The complex-plane RBF edge function

Each edge response is a sum of Gaussian bumps centred on a grid of complex
points $c_1,\dots,c_G$, combined with complex coefficients, plus a complex
linear base (residual):

$$
\psi_g(z) = \exp\!\Big(-\frac{|z - c_g|^2}{2 h^2}\Big), \qquad
y_o = \sum_{i}\sum_{g} W_{o,i,g}\,\psi_g(z_i) + (\mathbf{B} z)_o .
$$

The Gaussian acts on the **full** complex value (both quadratures), so the
learned edge function is genuinely two-dimensional rather than a real function
applied per part. The grid uses `num_grid**2` centres laid out on a square in
the complex plane (optionally learnable).

```python
import torch
import complextorch as ctorch

# A 2-layer complex KAN: 4 -> 8 -> 3 complex features.
model = ctorch.models.CVKAN([4, 8, 3], num_grid=8)
x = torch.randn(16, 4, dtype=torch.cfloat)
y = model(x)
print(y.shape, y.dtype)

# A single edge-function layer can fit a complex map such as z -> z**2.
layer = ctorch.nn.CVKANLayer(1, 1, num_grid=8)
```

See [CVKAN (arXiv:2502.02417)](https://arxiv.org/abs/2502.02417).
