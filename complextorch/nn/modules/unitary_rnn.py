r"""
Unitary Complex Recurrent Neural Networks
=========================================

Complex RNNs whose hidden-to-hidden transition is constrained to be **unitary**
(norm-preserving). A unitary recurrence has eigenvalues on the unit circle, so
repeated application neither shrinks nor blows up the hidden state -- the
classic complex-domain remedy for vanishing/exploding gradients on long
sequences.

The unitary matrix is produced by the **Cayley transform** of a learnable
skew-Hermitian generator:

.. math::

    A = M - M^H \;(\text{skew-Hermitian}), \qquad
    W = (I - A)(I + A)^{-1} \;(\text{unitary}),

for an unconstrained complex matrix :math:`M`. The recurrence then applies a
:class:`~complextorch.nn.AdaptiveModReLU` nonlinearity:

.. math::

    h_t = \sigma_{\text{modReLU}}(W h_{t-1} + V x_t).

References:

    - **Arjovsky, Shah, Bengio. Unitary Evolution Recurrent Neural Networks.**
      https://arxiv.org/abs/1511.06464
    - **Helfrich, Willmott, Ye. Orthogonal RNNs and the Cayley transform
      (scoRNN).** https://arxiv.org/abs/1707.09520
"""

import torch
import torch.nn as nn

from complextorch.nn import init
from complextorch.nn.modules.activation.split_type_B import AdaptiveModReLU
from complextorch.nn.modules.dropout import Dropout
from complextorch.nn.modules.linear import Linear

__all__ = ["UnitaryRNN", "UnitaryRNNCell"]


class UnitaryRNNCell(nn.Module):
    r"""
    Unitary Complex RNN Cell
    ------------------------

    One step of :math:`h_t = \sigma_{\text{modReLU}}(W h_{t-1} + V x_t)` with a
    unitary recurrence matrix :math:`W` (Cayley transform of a learnable
    skew-Hermitian generator) and a complex input projection :math:`V`.

    Note: the unitary matrix is materialised via a linear solve (an
    :math:`O(H^3)` op). :class:`UnitaryRNN` computes it once per forward and
    passes it back in through the ``w`` argument so it is not rebuilt every
    timestep; call :meth:`unitary_matrix` to fetch it directly.

    Args:
        input_size: feature size of ``x``.
        hidden_size: feature size of ``h``.
        bias: if ``True``, adds a bias to the input projection ``V``.
    """

    def __init__(self, input_size: int, hidden_size: int, bias: bool = True) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.V = Linear(input_size, hidden_size, bias=bias)
        self.M = nn.Parameter(torch.empty(hidden_size, hidden_size, dtype=torch.cfloat))
        self.register_buffer("eye", torch.eye(hidden_size, dtype=torch.cfloat))
        self.nonlinearity = AdaptiveModReLU(hidden_size)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Semi-unitary init for the skew-Hermitian generator.
        init.trabelsi_independent_(self.M)

    def unitary_matrix(self) -> torch.Tensor:
        r"""Materialise the unitary recurrence matrix :math:`W` via the Cayley
        transform of the skew-Hermitian generator :math:`A = M - M^H`."""
        A = self.M - self.M.conj().transpose(-2, -1)
        return torch.linalg.solve(self.eye + A, self.eye - A)

    def forward(
        self,
        input: torch.Tensor,
        hx: torch.Tensor | None = None,
        w: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if hx is None:
            hx = torch.zeros(
                input.shape[0],
                self.hidden_size,
                dtype=torch.cfloat,
                device=input.device,
            )
        # Reuse a precomputed unitary matrix when given (the sequence wrapper
        # materialises it once per forward instead of once per timestep).
        if w is None:
            w = self.unitary_matrix()
        # (W h)_i for row-vector batch h -> h @ W^T.
        pre = hx @ w.transpose(-2, -1) + self.V(input)
        return self.nonlinearity(pre)


class UnitaryRNN(nn.Module):
    r"""
    Multi-Layer Unitary Complex RNN
    -------------------------------

    Stacks :class:`UnitaryRNNCell` along the time axis. Unidirectional, with an
    API mirroring the relevant subset of :class:`torch.nn.RNN`
    (``num_layers``, ``batch_first``, ``dropout``).

    Args:
        input_size: feature size of ``x``.
        hidden_size: hidden feature size.
        num_layers: number of stacked layers.
        bias: if ``True``, adds biases to the input projections.
        batch_first: if ``True``, inputs/outputs are ``(B, T, F)`` instead of
            ``(T, B, F)``.
        dropout: inter-layer dropout probability (applied between layers when
            ``num_layers > 1``).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        batch_first: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first

        self.cells = nn.ModuleList(
            [
                UnitaryRNNCell(
                    input_size if layer == 0 else hidden_size, hidden_size, bias=bias
                )
                for layer in range(num_layers)
            ]
        )
        self.drop = Dropout(dropout) if dropout > 0 and num_layers > 1 else None

    def forward(
        self, input: torch.Tensor, hx: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.batch_first:
            input = input.transpose(0, 1)  # -> (T, B, F)
        seq_len, batch, _ = input.shape

        if hx is None:
            hx = torch.zeros(
                self.num_layers,
                batch,
                self.hidden_size,
                dtype=torch.cfloat,
                device=input.device,
            )

        outputs = input
        states: list[torch.Tensor] = []
        for layer, cell in enumerate(self.cells):
            h = hx[layer]
            # Materialise the unitary matrix once per layer, not per timestep.
            w = cell.unitary_matrix()
            outs: list[torch.Tensor] = []
            for t in range(seq_len):
                h = cell(outputs[t], h, w)
                outs.append(h)
            outputs = torch.stack(outs, dim=0)
            states.append(h)
            if self.drop is not None and layer < self.num_layers - 1:
                outputs = self.drop(outputs)

        if self.batch_first:
            outputs = outputs.transpose(0, 1)
        return outputs, torch.stack(states, dim=0)
