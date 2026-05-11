# Complex-valued activations

Complex-valued activation functions must take into account the two
degrees-of-freedom inherent to complex-valued data, typically represented as
real / imaginary parts or magnitude / phase. Two generalised classes of
activation operate on those respective representations and are defined as
*Type-A* and *Type-B* functions.

## Type-A — split on real / imaginary

Type-A activations consist of two real-valued functions, $G_\mathbb{R}(\cdot)$
and $G_\mathbb{I}(\cdot)$, applied to the real and imaginary parts of the input
tensor independently:

$$
G(\mathbf{z}) = G_\mathbb{R}(\mathbf{x}) + j\, G_\mathbb{I}(\mathbf{y})
$$

where $\mathbf{z} = \mathbf{x} + j\mathbf{y}$.

Under the hood, Type-A activations call
{func}`complextorch.nn.functional.apply_complex_split`. Examples in the
package: `CVSplitReLU`, `CVSplitTanh`, `CVSplitSigmoid`, `CELU`, `CCELU`,
`CGELU`. See the [activation reference](../api/complextorch/nn/modules/activation/index)
for the full list.

## Type-B — split on magnitude / phase

Type-B activations consist of two real-valued functions, $G_{||}(\cdot)$ and
$G_\angle(\cdot)$, applied to the magnitude (modulus) and phase (argument) of
the input tensor:

$$
G(\mathbf{z}) = G_{||}\!\left(|\mathbf{z}|\right) \,\exp\!\left(j\, G_\angle\!\left(\arg \mathbf{z}\right)\right).
$$

Type-B activations call
{func}`complextorch.nn.functional.apply_complex_polar`. Passing `phase_fun=None`
is an optimisation that skips the polar round-trip when the activation only
modifies magnitude. Examples: `modReLU`, `AdaptiveModReLU`, `CVPolarTanh`.

## Fully complex

Fully-complex activations fit neither the Type-A nor the Type-B
designation — they operate on the complex tensor directly. Use them when an
activation has a natural complex form (e.g., a learnable phase rotation).

## ReLU variants

A separate family generalises the rectified linear unit to the complex
plane:

- {class}`complextorch.nn.CReLU` / {class}`complextorch.nn.CVSplitReLU`
  (alias) — split Type-A: ReLU on real and imaginary parts independently.
- {class}`complextorch.nn.zReLU` — first-quadrant gating: passes $z$
  unchanged only when both $\Re z > 0$ and $\Im z > 0$.
- {class}`complextorch.nn.zAbsReLU` — magnitude threshold with a learnable
  cutoff; phase preserved.
- {class}`complextorch.nn.zLeakyReLU` — soft `zReLU` with a leaky slope
  outside the first quadrant.

## Phase-aware / manifold ReLUs

A third family operates on the magnitude / phase representation in a
**phase-aware** way (the operation depends on the phase, not just on each
component independently). These come from two paper lineages — CDS
([Singhal et al., CVPR 2022](https://arxiv.org/abs/2112.01525)) and SurReal
([Chakraborty, Xing, Yu, arxiv:1910.11334](https://arxiv.org/abs/1910.11334))
— and pair naturally with the symmetry-aware modules described in
[Co-domain symmetry](co-domain-symmetry.md).

- {class}`complextorch.nn.GTReLU` — learnable complex scaling
  $(\alpha + j\beta)\,z$ followed by an upper-half-plane phase mask
  $\theta \mapsto \theta \cdot \mathbf{1}[\theta \bmod 2\pi \in [0, \pi]]$
  with a custom autograd whose backward gradient is the mask itself.
  Optional learnable phase-rescaling.
- {class}`complextorch.nn.EquivariantPhaseReLU` — thresholds phase
  *relative to the channel-mean direction* so the operator commutes with
  any global phase rotation (strictly U(1)-equivariant).
- {class}`complextorch.nn.tReLU` — the tangent-space ReLU from SurReal
  Eq. 21-22:
  $r \mapsto \max(r, 1)$, $\arg z \mapsto \max(\arg z, 0)$.
  Parameter-free; the principled lift of standard ReLU onto the
  rotation+scaling manifold. Not equivariant by design (analogous to
  ReLU's lack of translation-equivariance on $\mathbb{R}$).
- {class}`complextorch.nn.wFMReLU` — learned affine on $\log|z|$ and
  $\arg z$ on the manifold; the port of RotLieNet's
  `manifoldReLUv2angle`, distinct from `tReLU`.

## When to use which

| Need | Reach for |
| --- | --- |
| Drop-in replacement for `nn.ReLU` / `nn.Tanh` | `CVSplitReLU` / `CVSplitTanh` (Type-A) |
| Preserve phase, modulate magnitude only | `modReLU`, `AdaptiveModReLU` (Type-B, `phase_fun=None`) |
| Phase-aware operation | Type-B with both `mag_fun` and `phase_fun` set |
| Strictly U(1)-equivariant ReLU | {class}`complextorch.nn.EquivariantPhaseReLU` |
| U(1)-invariant ReLU after a U(1)-invariant block | {class}`complextorch.nn.GTReLU` |
| Tangent-space ReLU on the manifold | {class}`complextorch.nn.tReLU` |
| Manifold-aware affine (paired with `wFMConv*`) | {class}`complextorch.nn.wFMReLU` |
| Learnable scalar phase shift | {class}`complextorch.nn.PhaseShift` |
| Learnable complex scaling | {class}`complextorch.nn.ComplexScaling` |
