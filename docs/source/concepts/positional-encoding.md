# Complex positional encodings

The native {class}`complextorch.nn.Transformer` and
{class}`complextorch.nn.MultiheadAttention` apply **no** positional encoding on
their own — attention is permutation-equivariant, so position has to be injected
explicitly. `complextorch.nn` ships three complex-valued schemes.

| Module | Type | Position | Mechanism |
| --- | --- | --- | --- |
| {class}`complextorch.nn.RotaryEmbedding` | relative | rotary (RoPE) | multiply Q/K by $e^{j\omega_k n}$ |
| {class}`complextorch.nn.SinusoidalPositionalEncoding` | absolute | additive | add $e^{j\omega_k n}$ |
| {class}`complextorch.nn.CoPE` | absolute | learnable | multiply by $e^{j(\omega_k n + \phi_k)}$ |

## Rotary embeddings (RoPE) are complex by construction

RoPE encodes position by **rotating** each feature channel by an angle
proportional to its position. Since this library's tensors are already complex,
a rotation is literally a multiplication by a unit phasor:

$$
\tilde{x}_{n,k} = x_{n,k}\, e^{j\omega_k n}, \qquad \omega_k = \text{base}^{-k/d}.
$$

The attention score uses the Hermitian inner product $Q\,K^H$, so the rotation
applied at query position $m$ and key position $n$ leaves a residual phase that
depends only on the **relative** offset $m-n$:

$$
\tilde{q}_{m,k}\,\overline{\tilde{k}_{n,k}}
   = q_{m,k}\,\overline{k_{n,k}}\; e^{j\omega_k (m-n)} .
$$

Apply it inside attention via the `rotary` argument (it is applied to the
per-head query/key tensors after projection, so build it with `dim=d_k`):

```python
import torch
import complextorch as ctorch

d_model, n_heads, d_head = 32, 4, 8
rope = ctorch.nn.RotaryEmbedding(dim=d_head)
mha = ctorch.nn.MultiheadAttention(n_heads, d_model, d_head, d_head, rotary=rope)

x = torch.randn(2, 16, d_model, dtype=torch.cfloat)   # (batch, length, d_model)
y = mha(x, x, x)
print(y.shape, y.dtype)
```

## Absolute encodings

{class}`complextorch.nn.SinusoidalPositionalEncoding` adds a fixed complex
sinusoidal phasor bank to the embeddings, and {class}`complextorch.nn.CoPE` is a
lightweight learnable variant (per-channel learnable frequency **and** phase,
`2·dim` parameters). The {class}`complextorch.models.ViT` exposes all three via
its `pos_encoding=` argument (`"learned"`, `"sinusoidal"`, `"rotary"`).
