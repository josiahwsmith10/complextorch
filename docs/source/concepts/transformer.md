# Complex transformer & attention masking

{class}`complextorch.nn.Transformer` mirrors {class}`torch.nn.Transformer` for
complex inputs: an encoder/decoder stack whose layers are built from
{class}`complextorch.nn.MultiheadAttention` and a feed-forward sub-block. As
of 2.2.0 the mirror is faithful on the two points that silently bite ports:

- **`batch_first=False` is the default**, matching `torch.nn.Transformer` —
  inputs are sequence-first `(T, B, F)` unless you pass `batch_first=True`.
- **The full mask API is supported** — `src_mask` / `tgt_mask` /
  `memory_mask` plus the three `*_key_padding_mask` arguments thread through
  the whole stack, so autoregressive decoding can actually be made causal.

```python
import torch
import complextorch as ctorch

model = ctorch.nn.Transformer(
    d_model=32, nhead=4, num_encoder_layers=2, num_decoder_layers=2,
    dim_feedforward=64, batch_first=True,
)

src = torch.randn(2, 10, 32, dtype=torch.cfloat)
tgt = torch.randn(2, 6, 32, dtype=torch.cfloat)

# Causal decoding: position t may not attend to positions > t.
tgt_mask = ctorch.nn.Transformer.generate_square_subsequent_mask(6)
out = model(src, tgt, tgt_mask=tgt_mask)
print(out.shape)
```

## Mask semantics

Every mask argument accepts either a **bool** tensor (`True` = disallowed /
padding) or an **additive float** tensor (`-inf` = disallowed; finite values
act as attention bias). Attention masks are `(L_q, L_k)` or broadcastable to
`(B, n_heads, L_q, L_k)`; key-padding masks are `(B, L)` **regardless of**
`batch_first`, exactly as in `torch.nn`. Masked positions receive exactly zero
attention weight, and a fully-masked row yields NaN — the same semantics as
`torch.nn.functional.softmax` over an all-`-inf` row.

## Masking under complex softmax

{class}`complextorch.nn.ScaledDotProductAttention` supports two score modes,
and "add `-inf` before the softmax" means something different in each:

- **`softmax_on="real"`** — the mask is added to $\Re(QK^H)$ before the
  ordinary real softmax.
- **`softmax_on="complex"`** — the mask is routed *into* the softmax module,
  because each variant must apply it to the real statistic it exponentiates:
  {class}`complextorch.nn.CVSoftMax` adds it to **both** the real and
  imaginary parts (else the imaginary-part softmax would still weight masked
  keys), while the polar variants ({class}`complextorch.nn.PhaseSoftMax`,
  {class}`complextorch.nn.MagSoftMax`) add it to the **magnitude logits** —
  adding `-inf` to the complex score itself would make $|z| = +\infty$ and
  hand the masked position *all* of the weight instead of none.

```{note}
A custom `SoftMaxClass` must accept `(input, mask)` to be used with
`attn_mask` under `softmax_on="complex"`; single-argument classes keep
working unmasked.
```

{class}`complextorch.nn.HolographicAttention` takes the same `attn_mask`
argument (its logits are real, so the mask is plain additive) — see
[Holographic attention](holographic-attention.md).

## Block structure

{class}`complextorch.nn.MultiheadAttention` is a *complete* post-norm
attention sub-block: QKV projections, the attention core, then an internal
residual connection and {class}`complextorch.nn.LayerNorm`. The transformer
encoder/decoder layers build on that directly (post-norm, matching
`torch.nn.Transformer`'s default). For **pre-norm** compositions, pass
`residual_norm=False` to get only the projected attention output and supply
your own residual — this is exactly how the pre-LN
{class}`complextorch.models.ViTLayer` composes it:

```python
nx = self.norm1(x)
x = x + self.attn(nx, nx, nx)   # attn built with residual_norm=False
```
