# Holographic (interference-aware) attention

{class}`complextorch.nn.HolographicAttention` is a drop-in alternative to
{class}`complextorch.nn.ScaledDotProductAttention` that treats attention as
**wave interference** rather than a phase-blind correlation. It is motivated by
signal-processing workloads (PolSAR, wireless channel prediction) where the
amplitude–phase coupling carries the signal.

It changes two things relative to standard scaled dot-product attention.

**1 · Phase-gated scores.** For a token pair $(i,j)$ with complex score
$s_{ij} = Q_i K_j^H$ and phase difference $\Delta\phi_{ij} = \angle s_{ij}$, the
magnitude-correlation similarity is gated by the phase discrepancy:

$$
\text{sim}_{ij} = \frac{\Re(s_{ij})}{\lVert Q_i\rVert\,\lVert K_j\rVert + \epsilon},
\qquad
W_{ij} = \frac{\text{sim}_{ij}}{\sqrt{d_k}}\, e^{-\alpha\lvert\Delta\phi_{ij}\rvert},
\qquad a_{ij} = \texttt{SoftMax}_j(W_{ij}).
$$

In-phase interactions are boosted; anti-phase ones are suppressed. The
discrepancy weight $\alpha \ge 0$ is **learnable**.

**2 · Coherent superposition.** Values are rotated by their phase offset before
the weighted sum, so aligned phases add constructively:

$$
H_i = \sum_j a_{ij}\, V_j\, e^{j\Delta\phi_{ij}} .
$$

```python
import torch
import complextorch as ctorch

mha = ctorch.nn.MultiheadAttention(
    n_heads=4, d_model=32, d_k=8, d_v=8, attention="holographic"
)
x = torch.randn(2, 16, 32, dtype=torch.cfloat)
y = mha(x, x, x)
print(y.shape, y.dtype)
```

## Guarding against phase collapse

The companion paper proves a phase-blind estimator has a non-trivial error
floor, and uses a dual-headed decoder (reconstruction + task) to force the model
to retain phase. Two helpers support that recipe:

- {class}`complextorch.nn.HolographicReconstructionLoss` — separate real/imag
  reconstruction term $\lVert\Re(\hat{x}-x)\rVert_2^2 + \lVert\Im(\hat{x}-x)\rVert_2^2$.
- {func}`complextorch.nn.phase_smoothness` — total-variation penalty on the
  wrapped phase difference between adjacent positions.

See [arXiv:2509.19331](https://arxiv.org/abs/2509.19331) for the full
formulation.
