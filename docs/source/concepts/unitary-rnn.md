# Unitary complex RNNs

{class}`complextorch.nn.UnitaryRNN` (and its cell
{class}`complextorch.nn.UnitaryRNNCell`) constrain the hidden-to-hidden
transition to be **unitary** — a norm-preserving recurrence whose eigenvalues
lie on the unit circle. Repeated application neither shrinks nor amplifies the
hidden state, which is the classic complex-domain remedy for vanishing /
exploding gradients on long sequences. This complements the existing
{class}`complextorch.nn.GRU` / {class}`complextorch.nn.LSTM` cells.

## The Cayley parameterisation

A unitary matrix is produced by the **Cayley transform** of a learnable
skew-Hermitian generator. For an unconstrained complex matrix $M$:

$$
A = M - M^H \quad(\text{skew-Hermitian}), \qquad
W = (I - A)(I + A)^{-1} \quad(\text{unitary}),
$$

and the recurrence applies an {class}`complextorch.nn.AdaptiveModReLU`
nonlinearity:

$$
h_t = \sigma_{\text{modReLU}}(W h_{t-1} + V x_t).
$$

The generator is initialised semi-unitarily with
{func}`complextorch.nn.init.trabelsi_independent_`.

```python
import torch
import complextorch as ctorch

cell = ctorch.nn.UnitaryRNNCell(input_size=8, hidden_size=16)

# The recurrence matrix is exactly unitary: W^H W = I, ||W h|| = ||h||.
W = cell.unitary_matrix()
h = torch.randn(4, 16, dtype=torch.cfloat)
print("norm preserved:",
      torch.allclose((h @ W.T).abs().pow(2).sum(-1), h.abs().pow(2).sum(-1), atol=1e-4))

rnn = ctorch.nn.UnitaryRNN(8, 16, num_layers=2, batch_first=True)
x = torch.randn(2, 50, 8, dtype=torch.cfloat)
out, h_n = rnn(x)
print(out.shape, h_n.shape)
```

```{note}
The unitary matrix is rematerialised via a linear solve on every step, so the
per-step cost is $O(H^3)$ — the price of an exactly-unitary transition. This is
a reference implementation; for very wide hidden states a parameterisation that
avoids the per-step solve would be faster.
```

See uRNN ([arXiv:1511.06464](https://arxiv.org/abs/1511.06464)) and the Cayley /
scoRNN orthogonal-RNN line ([arXiv:1707.09520](https://arxiv.org/abs/1707.09520)).
