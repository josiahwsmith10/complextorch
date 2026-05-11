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

A separate family — `zReLU`, `CReLU`, `zAbsReLU`, `zLeakyReLU` — generalises
the rectified linear unit to the complex plane. These are documented alongside
their classes in the API reference.

## When to use which

| Need | Reach for |
| --- | --- |
| Drop-in replacement for `nn.ReLU` / `nn.Tanh` | `CVSplitReLU` / `CVSplitTanh` (Type-A) |
| Preserve phase, modulate magnitude only | `modReLU`, `AdaptiveModReLU` (Type-B, `phase_fun=None`) |
| Phase-aware operation | Type-B with both `mag_fun` and `phase_fun` set |
| Learnable scalar phase shift | {class}`complextorch.nn.PhaseShift` |
