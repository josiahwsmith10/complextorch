# Complex state-space models (S4D / DSS / Mamba)

Structured state-space models (SSMs) are linear-time sequence models whose core
is a **diagonal complex** state transition. The complex-diagonal state is
exactly what gives these models their strength on long perceptual / signal
sequences, which makes them a natural fit for this library and for the long 1-D
signals it targets.

All of the layers operate on complex sequences of shape `(B, L, H)` — batch,
length, channels.

## The diagonal SSM

A per-channel single-input/single-output SSM with complex state
$x \in \mathbb{C}^N$ evolves as

$$
x'(t) = A\,x(t) + B\,u(t), \qquad y(t) = C\,x(t) + D\,u(t),
$$

with **diagonal** $A \in \mathbb{C}^N$. Discretising with a per-channel step
$\Delta$ (zero-order hold) gives $\bar A = e^{\Delta A}$,
$\bar B = (\bar A - 1)A^{-1}B$, and a causal convolution kernel

$$
\bar K_\ell = \sum_n C_n\, \bar A_n^{\ell}\, \bar B_n, \qquad y = u * \bar K + D\,u.
$$

{class}`complextorch.nn.S4D` materialises this kernel and applies it with an FFT
long-convolution during training; {meth}`recurrence` runs the mathematically
equivalent step-by-step recurrence for exact streaming inference. The diagonal
$A$ is parameterised with a negative real part for stability and initialised to
the S4D-Lin schedule $A_n = -\tfrac12 + j\pi n$.

{class}`complextorch.nn.DSS` is a sibling using the Diagonal-State-Space kernel
normalisation, which bounds the kernel over the sequence length regardless of
the sign of $\Re(A)$. {class}`complextorch.nn.S4DBlock` wraps either variant in
a residual `RMSNorm → SSM → GELU → Linear` block for stacking.

```python
import torch
import complextorch as ctorch

block = ctorch.nn.S4DBlock(channels=16, state_size=64)
u = torch.randn(2, 128, 16, dtype=torch.cfloat)   # long 1-D signal
y = block(u)
print(y.shape, y.dtype)

# FFT convolution and the recurrent rollout agree:
ssm = ctorch.nn.S4D(channels=16, state_size=64)
torch.testing.assert_close(ssm(u), ssm.recurrence(u), atol=1e-4, rtol=1e-4)
```

## Selective (Mamba) variant

{class}`complextorch.nn.MambaBlock` makes the SSM **selective**: $B$, $C$ and
the step $\Delta$ become input-dependent, so the model can choose what to
propagate or forget. Because the dynamics are time-varying there is no global
convolution kernel — the state is advanced with a sequential selective scan
(pure-torch; no custom kernel).

```python
mamba = ctorch.nn.MambaBlock(channels=16, state_size=16)
y = mamba(u)
print(y.shape, y.dtype)
```

See S4 ([arXiv:2111.00396](https://arxiv.org/abs/2111.00396)), S4D
([arXiv:2206.11893](https://arxiv.org/abs/2206.11893)), DSS
([arXiv:2203.14343](https://arxiv.org/abs/2203.14343)), and Mamba
([arXiv:2312.00752](https://arxiv.org/abs/2312.00752)).
