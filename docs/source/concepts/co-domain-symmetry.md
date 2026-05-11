# Co-domain symmetry and U(1)-equivariant networks

Complex-valued data often has a meaningful **co-domain symmetry**: the
overall phase of a signal is arbitrary (the absolute time reference of a
radar pulse, the global phase of an MR coil, the carrier phase of a comms
symbol), so a sensible model should either ignore that overall phase
(**U(1)-invariance**) or rotate its features along with the input
(**U(1)-equivariance**) — and never silently break it.

`complextorch.nn` provides a coherent set of building blocks for both
regimes. They originate from two papers:

- **SurReal** (Chakraborty, Xing, Yu — [arxiv:1910.11334](https://arxiv.org/abs/1910.11334)) —
  treats $\mathbb{C} \setminus \{0\}$ as the Riemannian manifold
  $\mathbb{R}^+ \times SO(2)$ and defines a weighted-Fréchet-mean (wFM)
  convolution and a tangent-space ReLU on it.
- **Co-Domain Symmetry (CDS)** (Singhal, Xing, Yu, CVPR 2022) — builds
  Invariant ("I-type") and Equivariant ("E-type") networks via
  phase-modulation layers, phase-thresholding activations, and a
  magnitude-only batch norm.

## What's "equivariant" and "invariant"?

For a global phase rotation by $\psi$ — i.e. $x \mapsto e^{j\psi}\,x$ — an
operator $M(\cdot)$ is

- **U(1)-equivariant** if $M(e^{j\psi}\,x) = e^{j\psi}\,M(x)$ — output
  rotates the same way the input does.
- **U(1)-invariant** if $M(e^{j\psi}\,x) = M(x)$ — output is unchanged.

A network is *equivariant up to its classifier* if every layer is
equivariant; a *fully invariant* network ends with an invariant head
(typically a distance-to-prototypes or a magnitude-only readout).

The headline correctness tests for each module live in
`tests/invariants/test_equivariance.py`.

## Module map

### Strictly U(1)-equivariant

| Module | Operation |
| --- | --- |
| {class}`complextorch.nn.PhaseShift` | $y = e^{j\phi}\,z$ |
| {class}`complextorch.nn.ComplexScaling` | $y = (\alpha + j\beta)\,z$ |
| {class}`complextorch.nn.MagBatchNorm1d` / `2d` / `3d` | $y = z \cdot \mathrm{BN}(|z|) / (|z| + \varepsilon)$ |
| {class}`complextorch.nn.EquivariantPhaseReLU` | channel-mean-relative phase mask |
| {class}`complextorch.nn.wFMConvStrict2d` | SurReal Eq. 14-16, convex weights ($\sum w = 1, w \geq 0$) |

### Strictly U(1)-invariant

| Module | Operation |
| --- | --- |
| {class}`complextorch.nn.PhaseDivConv1d` / `2d` / `3d` | $y = x \cdot \overline{g(x)} / (|g(x)|^2 + \varepsilon)$ |
| {class}`complextorch.nn.PhaseConjConv1d` / `2d` / `3d` | $y = x \cdot \overline{g(x)}$ (invariant when $g$ is C-linear) |
| {class}`complextorch.nn.PrototypeDistance` (E-type call) | logits unchanged when both $z$ and the reference rotate by $e^{j\psi}$ |

### Tangent-space / manifold operators

| Module | Operation | Property |
| --- | --- | --- |
| {class}`complextorch.nn.tReLU` | $r \mapsto \max(r, 1)$, $\arg z \mapsto \max(\arg z, 0)$ | SurReal Eq. 21-22 — not equivariant by design, analogous to standard ReLU's lack of translation-equivariance |
| {class}`complextorch.nn.wFMConv1d` / `wFMConv2d` | port of RotLieNet `ComplexConv2Deffgroup` | approximate, with non-paper pre-modulation; prefer `wFMConvStrict2d` for paper-faithful math |
| {class}`complextorch.nn.wFMReLU` | learned affine on $\log|z|$ and $\arg z$ | port of `manifoldReLUv2angle`; not paper Eq. 21-22 |
| {class}`complextorch.nn.wFMDistanceLinear` | distance-to-Fréchet-mean head | output is **real** (invariants for classification) |

## Composing an equivariant block

Any composition of strictly U(1)-equivariant ops is itself
U(1)-equivariant. For example,

```python
import complextorch.nn as cnn

equivariant_block = nn.Sequential(
    cnn.wFMConvStrict2d(in_channels, out_channels, kernel_size=3, padding=0),
    cnn.ComplexScaling(out_channels),
    cnn.EquivariantPhaseReLU(out_channels),
    cnn.MagBatchNorm2d(out_channels),
)
```

To make the *whole network* invariant for classification, follow such a
stack with a {class}`complextorch.nn.PrototypeDistance` head and pass a
reference vector (typically a sum-pool of the features) so that
prototypes co-rotate with the input:

```python
features = backbone(x)                           # equivariant
z = features.flatten(2).mean(dim=2)              # [B, C], still equivariant
ref = features.sum(dim=1, keepdim=False).mean()  # [B, 1], rotates with x
logits = head(z, reference=ref)                  # invariant
```

This is exactly the pattern used by
{class}`complextorch.models.CDSEquivariant`.

## Where invariance comes from

- **`PhaseDivConv`** uses the fact that a global phase rotation cancels
  in numerator and denominator: $(e^{j\psi}\,x) \cdot \overline{e^{j\psi}\,g(x)} / |e^{j\psi}\,g(x)|^2 = x \cdot \overline{g(x)} / |g(x)|^2$.
- **`PhaseConjConv`** is invariant for the same reason when $g$ is
  complex-linear (the cfloat-native default). The CDS paper described it
  as "phase-mixing" because the reference code decomposed $g$ as two
  real convs (which still happens to be C-linear, so the same cancellation
  applies; the "phase-mixing" framing was loose).
- **`PrototypeDistance` with `reference=`**: when both $z$ and the
  reference rotate by $e^{j\psi}$, the per-prototype distances
  $|z - y \cdot p_k|$ are unchanged, so logits are invariant.

## Caveats

- **Padding breaks equivariance.** `nn.Unfold(padding > 0)` zero-pads
  reals/phases independently. A zero magnitude cannot be rotated
  meaningfully — the $(\log|0|, \arg 0)$ representation is ambiguous.
  `wFMConvStrict2d` is strictly equivariant only with `padding=0`; with
  positive padding, boundary positions degrade. The docstring carries
  the full note.
- **`tReLU` is not equivariant.** This mirrors standard ReLU not being
  translation-equivariant in real-valued nets — the tangent-space ReLU
  is a principled lift, not a free lunch.
- **`wFMConv2d` (existing) ≠ `wFMConvStrict2d` (new).** The former is
  the port of RotLieNet's `ComplexConv2Deffgroup`, with a non-paper
  pre-modulation and `fold(unfold(·))` smear; the latter is the
  paper-faithful Eq. 14-16 implementation. Prefer the strict variant
  when you need provable equivariance.
