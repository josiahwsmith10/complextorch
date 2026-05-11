r"""
Complex-Valued Recurrent Neural Networks
========================================

Drop-in complex analogues of :class:`torch.nn.GRUCell`, :class:`torch.nn.GRU`,
:class:`torch.nn.LSTMCell`, and :class:`torch.nn.LSTM`.

Cells (:class:`GRUCell`, :class:`LSTMCell`) are built from complex
:class:`Linear` layers and complex activations and are mathematically the
standard cell equations applied to complex inputs and states. Multi-layer
sequence wrappers (:class:`GRU`, :class:`LSTM`) stack cells along the time
axis — they do not use PyTorch's CuDNN-fused real RNN under the hood (which
would need a parameterization trick that subtly differs from the cell math).

Each cell accepts ``batchnorm=False``; setting it to ``True`` inserts a
:class:`BatchNorm1d` after every internal linear projection (analogous to
*Recurrent Batch Normalization* (Cooijmans et al., 2017) for the complex case).
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from complextorch.nn.modules.activation.split_type_A import CTanh, CSigmoid
from complextorch.nn.modules.batchnorm import BatchNorm1d
from complextorch.nn.modules.dropout import Dropout
from complextorch.nn.modules.linear import Linear

__all__ = ["GRUCell", "GRU", "LSTMCell", "LSTM"]


def _maybe_bn(features: int, enabled: bool) -> nn.Module:
    return BatchNorm1d(features) if enabled else nn.Identity()


class GRUCell(nn.Module):
    r"""
    Complex-Valued GRU Cell
    -----------------------

    Standard GRU equations applied to a complex input and hidden state:

    .. math::

        r_t &= \sigma(W_{ir} x_t + W_{hr} h_{t-1} + b_r) \\
        z_t &= \sigma(W_{iz} x_t + W_{hz} h_{t-1} + b_z) \\
        n_t &= \tanh(W_{in} x_t + r_t \odot (W_{hn} h_{t-1}) + b_n) \\
        h_t &= (1 - z_t) \odot n_t + z_t \odot h_{t-1}

    where :math:`\sigma` and :math:`\tanh` are the split (Type-A) complex
    activations :class:`CSigmoid` / :class:`CTanh`, and all weights/biases
    are complex.

    Args:
        input_size: feature size of ``x``.
        hidden_size: feature size of ``h``.
        bias: if ``True``, adds biases to the projections.
        batchnorm: if ``True``, wraps each linear projection in
            :class:`BatchNorm1d`. Useful for stabilizing deep recurrent stacks.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        batchnorm: bool = False,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batchnorm = batchnorm

        # 6 linear projections: 3 input (r, z, n) + 3 hidden (r, z, n).
        self.w_ir = Linear(input_size, hidden_size, bias=bias)
        self.w_iz = Linear(input_size, hidden_size, bias=bias)
        self.w_in = Linear(input_size, hidden_size, bias=bias)
        self.w_hr = Linear(hidden_size, hidden_size, bias=bias)
        self.w_hz = Linear(hidden_size, hidden_size, bias=bias)
        self.w_hn = Linear(hidden_size, hidden_size, bias=bias)

        if batchnorm:
            self.bn_ir = BatchNorm1d(hidden_size)
            self.bn_iz = BatchNorm1d(hidden_size)
            self.bn_in = BatchNorm1d(hidden_size)
            self.bn_hr = BatchNorm1d(hidden_size)
            self.bn_hz = BatchNorm1d(hidden_size)
            self.bn_hn = BatchNorm1d(hidden_size)
        self.sigmoid = CSigmoid()
        self.tanh = CTanh()

    def _bn(self, name: str, x: torch.Tensor) -> torch.Tensor:
        if not self.batchnorm:
            return x
        # BatchNorm1d expects (B, C); use directly.
        return getattr(self, name)(x)

    def forward(
        self, input: torch.Tensor, hx: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if hx is None:
            hx = torch.zeros(
                input.shape[0], self.hidden_size, dtype=input.dtype, device=input.device
            )
        r = self.sigmoid(
            self._bn("bn_ir", self.w_ir(input)) + self._bn("bn_hr", self.w_hr(hx))
        )
        z = self.sigmoid(
            self._bn("bn_iz", self.w_iz(input)) + self._bn("bn_hz", self.w_hz(hx))
        )
        n = self.tanh(
            self._bn("bn_in", self.w_in(input)) + r * self._bn("bn_hn", self.w_hn(hx))
        )
        return (1 - z) * n + z * hx


class LSTMCell(nn.Module):
    r"""
    Complex-Valued LSTM Cell
    ------------------------

    Standard LSTM equations applied to complex inputs and states:

    .. math::

        i_t &= \sigma(W_{ii} x_t + W_{hi} h_{t-1} + b_i) \\
        f_t &= \sigma(W_{if} x_t + W_{hf} h_{t-1} + b_f) \\
        g_t &= \tanh(W_{ig} x_t + W_{hg} h_{t-1} + b_g) \\
        o_t &= \sigma(W_{io} x_t + W_{ho} h_{t-1} + b_o) \\
        c_t &= f_t \odot c_{t-1} + i_t \odot g_t \\
        h_t &= o_t \odot \tanh(c_t)

    Args:
        input_size: feature size of ``x``.
        hidden_size: feature size of ``h`` and ``c``.
        bias: if ``True``, adds biases to the projections.
        batchnorm: if ``True``, wraps each linear projection in
            :class:`BatchNorm1d`.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        batchnorm: bool = False,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batchnorm = batchnorm

        self.w_ii = Linear(input_size, hidden_size, bias=bias)
        self.w_if = Linear(input_size, hidden_size, bias=bias)
        self.w_ig = Linear(input_size, hidden_size, bias=bias)
        self.w_io = Linear(input_size, hidden_size, bias=bias)
        self.w_hi = Linear(hidden_size, hidden_size, bias=bias)
        self.w_hf = Linear(hidden_size, hidden_size, bias=bias)
        self.w_hg = Linear(hidden_size, hidden_size, bias=bias)
        self.w_ho = Linear(hidden_size, hidden_size, bias=bias)

        if batchnorm:
            for gate in ("ii", "if_", "ig", "io", "hi", "hf", "hg", "ho"):
                setattr(self, f"bn_{gate}", BatchNorm1d(hidden_size))
        self.sigmoid = CSigmoid()
        self.tanh = CTanh()

    def _bn(self, name: str, x: torch.Tensor) -> torch.Tensor:
        if not self.batchnorm:
            return x
        return getattr(self, name)(x)

    def forward(
        self,
        input: torch.Tensor,
        hx: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if hx is None:
            zero_h = torch.zeros(
                input.shape[0], self.hidden_size, dtype=input.dtype, device=input.device
            )
            zero_c = torch.zeros_like(zero_h)
            h_prev, c_prev = zero_h, zero_c
        else:
            h_prev, c_prev = hx
        i = self.sigmoid(
            self._bn("bn_ii", self.w_ii(input)) + self._bn("bn_hi", self.w_hi(h_prev))
        )
        # ``if`` is a Python keyword, so the attribute is named ``w_if`` and
        # the batch-norm is ``bn_if_``.
        f = self.sigmoid(
            self._bn("bn_if_", self.w_if(input)) + self._bn("bn_hf", self.w_hf(h_prev))
        )
        g = self.tanh(
            self._bn("bn_ig", self.w_ig(input)) + self._bn("bn_hg", self.w_hg(h_prev))
        )
        o = self.sigmoid(
            self._bn("bn_io", self.w_io(input)) + self._bn("bn_ho", self.w_ho(h_prev))
        )
        c = f * c_prev + i * g
        h = o * self.tanh(c)
        return h, c


# ---------------------------------------------------------------------------
# Multi-layer sequence wrappers
# ---------------------------------------------------------------------------


class _RNNBase(nn.Module):
    """Internal base for multi-layer cell-stacking RNNs."""

    _cell_class = GRUCell

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        batch_first: bool = False,
        dropout: float = 0.0,
        bidirectional: bool = False,
        batchnorm: bool = False,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        cells_fwd: List[nn.Module] = []
        cells_bwd: List[nn.Module] = []
        for layer in range(num_layers):
            in_size = input_size if layer == 0 else hidden_size * self.num_directions
            cells_fwd.append(
                self._cell_class(in_size, hidden_size, bias=bias, batchnorm=batchnorm)
            )
            if bidirectional:
                cells_bwd.append(
                    self._cell_class(
                        in_size, hidden_size, bias=bias, batchnorm=batchnorm
                    )
                )
        self.cells_fwd = nn.ModuleList(cells_fwd)
        self.cells_bwd = nn.ModuleList(cells_bwd) if bidirectional else None

        self.drop = Dropout(dropout) if dropout > 0 and num_layers > 1 else None


class GRU(_RNNBase):
    r"""
    Multi-Layer Complex-Valued GRU
    ------------------------------

    Stacks :class:`GRUCell` along the time axis. Compatible API with
    :class:`torch.nn.GRU` (``num_layers``, ``batch_first``, ``dropout``,
    ``bidirectional``).

    Note: this implementation rolls along time in Python rather than using
    the (real-only) CuDNN fused kernel, so expect a per-step Python overhead
    on long sequences.
    """

    _cell_class = GRUCell

    def forward(
        self,
        input: torch.Tensor,
        hx: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.batch_first:
            input = input.transpose(0, 1)  # -> (T, B, F)
        seq_len, batch, _ = input.shape

        if hx is None:
            hx = torch.zeros(
                self.num_layers * self.num_directions,
                batch,
                self.hidden_size,
                dtype=input.dtype,
                device=input.device,
            )

        outputs = input
        layer_states: List[torch.Tensor] = []

        for layer in range(self.num_layers):
            fwd_cell = self.cells_fwd[layer]
            h_f = hx[layer * self.num_directions]
            outs_f: List[torch.Tensor] = []
            for t in range(seq_len):
                h_f = fwd_cell(outputs[t], h_f)
                outs_f.append(h_f)
            out_f = torch.stack(outs_f, dim=0)

            if self.bidirectional:
                bwd_cell = self.cells_bwd[layer]
                h_b = hx[layer * self.num_directions + 1]
                outs_b: List[torch.Tensor] = []
                for t in range(seq_len - 1, -1, -1):
                    h_b = bwd_cell(outputs[t], h_b)
                    outs_b.append(h_b)
                outs_b.reverse()
                out_b = torch.stack(outs_b, dim=0)
                outputs = torch.cat([out_f, out_b], dim=-1)
                layer_states.append(h_f)
                layer_states.append(h_b)
            else:
                outputs = out_f
                layer_states.append(h_f)

            if self.drop is not None and layer < self.num_layers - 1:
                outputs = self.drop(outputs)

        if self.batch_first:
            outputs = outputs.transpose(0, 1)
        new_hx = torch.stack(layer_states, dim=0)
        return outputs, new_hx


class LSTM(_RNNBase):
    r"""
    Multi-Layer Complex-Valued LSTM
    -------------------------------

    Stacks :class:`LSTMCell` along the time axis. Compatible API with
    :class:`torch.nn.LSTM`.
    """

    _cell_class = LSTMCell

    def forward(
        self,
        input: torch.Tensor,
        hx: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if self.batch_first:
            input = input.transpose(0, 1)
        seq_len, batch, _ = input.shape

        n_dir = self.num_directions
        if hx is None:
            shape = (self.num_layers * n_dir, batch, self.hidden_size)
            h0 = torch.zeros(shape, dtype=input.dtype, device=input.device)
            c0 = torch.zeros_like(h0)
        else:
            h0, c0 = hx

        outputs = input
        new_h: List[torch.Tensor] = []
        new_c: List[torch.Tensor] = []

        for layer in range(self.num_layers):
            fwd_cell = self.cells_fwd[layer]
            h_f, c_f = h0[layer * n_dir], c0[layer * n_dir]
            outs_f: List[torch.Tensor] = []
            for t in range(seq_len):
                h_f, c_f = fwd_cell(outputs[t], (h_f, c_f))
                outs_f.append(h_f)
            out_f = torch.stack(outs_f, dim=0)

            if self.bidirectional:
                bwd_cell = self.cells_bwd[layer]
                h_b, c_b = h0[layer * n_dir + 1], c0[layer * n_dir + 1]
                outs_b: List[torch.Tensor] = []
                for t in range(seq_len - 1, -1, -1):
                    h_b, c_b = bwd_cell(outputs[t], (h_b, c_b))
                    outs_b.append(h_b)
                outs_b.reverse()
                out_b = torch.stack(outs_b, dim=0)
                outputs = torch.cat([out_f, out_b], dim=-1)
                new_h.extend([h_f, h_b])
                new_c.extend([c_f, c_b])
            else:
                outputs = out_f
                new_h.append(h_f)
                new_c.append(c_f)

            if self.drop is not None and layer < self.num_layers - 1:
                outputs = self.drop(outputs)

        if self.batch_first:
            outputs = outputs.transpose(0, 1)
        return outputs, (torch.stack(new_h, dim=0), torch.stack(new_c, dim=0))
