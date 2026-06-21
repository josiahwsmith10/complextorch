r"""
Complex-Valued Kolmogorov-Arnold Networks (CVKAN)
=================================================

Kolmogorov-Arnold Networks replace the fixed weights of an MLP with **learnable
univariate basis functions** on the edges. :class:`CVKANLayer` is a complex
adaptation that places a learnable radial-basis expansion over the **complex
plane**: each edge response is a sum of Gaussian bumps centred on a grid of
complex points, combined with complex coefficients, plus a complex linear base
(residual) term.

For a complex input feature :math:`z` and grid centres
:math:`c_1,\dots,c_G \in \mathbb{C}`:

.. math::

    \psi_g(z) = \exp\!\Big(-\frac{|z - c_g|^2}{2h^2}\Big), \qquad
    y_o = \sum_{i} \sum_{g} W_{o,i,g}\, \psi_g(z_i) + (\mathbf{B} z)_o,

with complex coefficients :math:`W` and a complex linear base :math:`\mathbf{B}`.
The Gaussian acts on the full complex value (both quadratures), so the learned
edge function is genuinely two-dimensional.

Reference:

    - **CVKAN: Complex-Valued Kolmogorov-Arnold Networks.**
      https://arxiv.org/abs/2502.02417
"""

import torch
import torch.nn as nn

from complextorch.nn.modules.linear import Linear

__all__ = ["CVKANLayer"]


class CVKANLayer(nn.Module):
    r"""
    Complex-Valued KAN Layer
    ------------------------

    Maps ``in_features`` complex inputs to ``out_features`` complex outputs with
    a learnable complex-plane radial-basis expansion plus a complex linear base.

    Args:
        in_features: number of input features.
        out_features: number of output features.
        num_grid: grid resolution **per axis**; the basis uses ``num_grid**2``
            complex centres laid out on a square grid in the complex plane.
        grid_range: ``(low, high)`` extent of the grid on each axis.
        learnable_grid: if ``True`` the centre locations are learnable.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_grid: int = 8,
        grid_range: tuple[float, float] = (-1.0, 1.0),
        learnable_grid: bool = False,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_grid = num_grid

        axis = torch.linspace(grid_range[0], grid_range[1], num_grid)
        re, im = torch.meshgrid(axis, axis, indexing="ij")
        centers = torch.complex(re.reshape(-1), im.reshape(-1))  # (G,), G=num_grid^2
        self.num_centers = centers.numel()
        if learnable_grid:
            self.centers = nn.Parameter(centers)
        else:
            self.register_buffer("centers", centers)

        # RBF bandwidth = grid spacing.
        self.h = (grid_range[1] - grid_range[0]) / max(num_grid - 1, 1)

        self.spline_weight = nn.Parameter(
            0.1
            * torch.randn(
                out_features, in_features, self.num_centers, dtype=torch.cfloat
            )
        )
        self.base = Linear(in_features, out_features)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        r"""Apply the complex KAN edge functions.

        Args:
            input (torch.Tensor): complex ``(..., in_features)`` tensor.

        Returns:
            torch.Tensor: complex ``(..., out_features)`` tensor.
        """
        diff = input.unsqueeze(-1) - self.centers  # (..., in, G)
        psi = torch.exp(-(diff.abs() ** 2) / (2 * self.h**2))  # (..., in, G), real
        spline = torch.einsum(
            "...ig,oig->...o", psi.to(self.spline_weight.dtype), self.spline_weight
        )
        return spline + self.base(input)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"num_grid={self.num_grid}"
        )
