# Steinmetz & Analytic networks

A different way to process complex data: instead of complex-native layers, use
**parallel real-valued subnetworks** whose outputs are coupled into a complex
latent. {class}`complextorch.models.SteinmetzNetwork` implements this multi-view
approach, and {class}`complextorch.models.AnalyticNeuralNetwork` adds a
consistency penalty that provably tightens the generalisation-gap bound.

## Steinmetz network

The stacked real/imag features feed two parallel real MLPs whose outputs become
the real and imaginary parts of the complex output:

$$
u = f_\Re([\Re z, \Im z]), \quad v = f_\Im([\Re z, \Im z]), \quad \hat z = u + j v.
$$

```python
import torch
import complextorch as ctorch

net = ctorch.models.SteinmetzNetwork(in_features=4, hidden_features=32, out_features=8)
x = torch.randn(16, 4, dtype=torch.cfloat)
y = net(x)
print(y.shape, y.dtype)
```

## Analytic network: the consistency penalty

The Analytic Neural Network adds the **analytic-signal consistency penalty**
({class}`complextorch.nn.AnalyticSignalLoss`), which drives the imaginary part of
the latent towards the Hilbert transform of its real part:

$$
\mathcal{L}_{\text{analytic}}(\hat z) =
   \big\| \Im(\hat z) - \mathcal{H}\{\Re(\hat z)\} \big\|^2 .
$$

Enforcing this orthogonal real/imag relationship is what lowers the
generalisation bound relative to a generic Steinmetz network.

```python
net = ctorch.models.AnalyticNeuralNetwork(4, 32, 8)
out = net(x)
loss = out.abs().pow(2).mean() + 0.1 * net.consistency_loss(out)  # task + consistency
```

The Hilbert transform / analytic signal used here is available directly as
{func}`complextorch.signal.hilbert` / {func}`complextorch.signal.analytic_signal`.

See [Steinmetz Neural Networks (arXiv:2409.10075)](https://arxiv.org/abs/2409.10075).
