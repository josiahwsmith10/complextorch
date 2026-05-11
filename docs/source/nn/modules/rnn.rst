Recurrent Neural Networks
=========================

Complex-valued cells and multi-layer sequence wrappers (GRU / LSTM). Cells
are built from complex :class:`Linear` projections plus complex split
activations; the multi-layer wrappers stack cells along the time axis.

Each cell accepts ``batchnorm=False``; setting it to ``True`` inserts a
:class:`BatchNorm1d` after every internal linear projection.

.. automodule:: complextorch.nn.modules.rnn
    :members:
