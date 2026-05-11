# Changelog

All notable changes to `complextorch` are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0]

### Added

- New top-level subpackages: `complextorch.signal` (`pwelch`),
  `complextorch.transforms` (torchcvnn-style dataloader transforms —
  `LogAmplitude`, `FFT2`, `IFFT2`, `FFTResize`, `PolSAR`, `Normalize`,
  `RandomPhase`, …), `complextorch.datasets` (SAR / MRI dataset surface;
  `SAMPLE` and `SLCDataset` are full implementations, the SAR/MRI-specific
  readers are present as importable stubs with upstream pointers), and
  `complextorch.models` (Vision Transformer with `vit_t/s/b/l/h` presets).
- `complextorch.nn.init`: `kaiming_normal_`, `kaiming_uniform_`,
  `xavier_normal_`, `xavier_uniform_`, `trabelsi_standard_`,
  `trabelsi_independent_` — variance-correct complex weight initialisers.
  (PyTorch's built-ins treat real and imaginary parts independently, which is
  wrong for complex magnitude.)
- `complextorch.nn.relevance` (complex Variational Dropout & Automatic
  Relevance Determination) and `complextorch.nn.masked` (fixed-mask
  sparsified layers) subsystems for learned-sparsity workflows. Adds
  `LinearVD`, `LinearARD`, `BilinearVD/ARD`, `Conv{1,2,3}dVD/ARD`,
  `LinearMasked`/`Conv*dMasked`, plus the deploy/extract helpers
  `named_penalties`, `compute_ard_masks`, `deploy_masks`. Requires `scipy`
  (new runtime dependency).
- RNN family: `GRUCell`, `GRU`, `LSTMCell`, `LSTM` (cell-based, with
  optional `batchnorm=True` for stable deep stacks).
- Transformer family: `TransformerEncoderLayer`, `TransformerEncoder`,
  `TransformerDecoderLayer`, `TransformerDecoder`, `Transformer`.
- Normalisation: `RMSNorm`, `GroupNorm`, `NaiveBatchNorm{1,2,3}d`
  (split-form baseline). The functional whitening helpers
  (`whiten2x2_batch_norm`, `whiten2x2_layer_norm`, `inv_sqrtm2x2`,
  `batch_norm`, `layer_norm`) are now public in
  `complextorch.nn.functional`.
- Pooling: `MagMaxPool{1,2,3}d` (magnitude-argmax, the canonical complex
  max-pool — `torch.nn.MaxPool*d` doesn't define `>` on complex),
  `AvgPool{1,2,3}d`.
- Channel dropout: `Dropout1d`, `Dropout2d`, `Dropout3d` with shared
  real/imag mask (Trabelsi 2018).
- Upsampling: `Upsample` (split real/imag) and `PolarUpsample`
  (phase-preserving polar form).
- Activations: `CELU`, `CCELU`, `CGELU` (split-type-A ELU/CELU/GELU +
  `CVSplit*` aliases), `zAbsReLU`, `zLeakyReLU` (first-quadrant + leaky
  variants), `Mod` (magnitude as module), `AdaptiveModReLU` (per-channel
  learnable threshold). Existing `modReLU` gains a `learnable=True` flag for
  a scalar trainable threshold.
- Layers: `Bilinear` (with `conjugate=True/False`), `InterleavedToComplex` /
  `ComplexToInterleaved` / `ConcatenatedToComplex` /
  `ComplexToConcatenated` / `RealToComplex` (layout-conversion modules),
  `PhaseShift` (learnable per-channel phase rotation).
- Loss: `MSELoss` matching `torch.nn.MSELoss` exactly (no 1/2 factor —
  distinct from `CVQuadError`).
- Optional dependencies gated behind extras: `complextorch[datasets]` pulls
  in `h5py`; `complextorch[datasets-alos]` pulls in `rasterio`.
- Comprehensive test suite under `tests/`, mirroring the `complextorch/`
  tree 1:1 (~490 tests). Covers every public class and helper, including
  Fast/Slow numerical equivalence (state-dict-aligned weights), full loss
  reduction matrix + invalid-reduction checks, Hypothesis-driven round-trip
  invariants (polar, casting, FFT), `scipy.special.expi` parity +
  `gradcheck` for `_expi`, and a parameterized sweep over the 11 dataset
  stubs.
- `[test]` extras now pull in `pytest-xdist` (parallel runs via `-n auto`)
  and `hypothesis` (property tests).

### Changed

- **BREAKING:** `MultiheadAttention` / `ScaledDotProductAttention` now use
  the Hermitian inner product `QKᴴ` (was `QKᵀ` — a math bug). New
  `softmax_on='complex'|'real'` flag selects the attention-weight semantics;
  default `'complex'` keeps the existing `CVSoftMax` behaviour.
- **BREAKING:** `Linear` / `SlowLinear` / fast `Conv{1,2,3}d` / fast
  `ConvTranspose{1,2,3}d` default `bias=True` to match `torch.nn`. Pass
  `bias=False` explicitly if you relied on the old default.
- CI enforces `--cov-fail-under=100` on Python 3.10 / 3.11 / 3.12 — any PR
  that drops line coverage fails automatically. Coverage config (omit list,
  `exclude_lines` for `raise NotImplementedError` / `pragma: no cover` /
  `if TYPE_CHECKING:` / `@overload`) lives in `pyproject.toml`.
- Documentation migrated to PyData Sphinx Theme + MyST + sphinx-autoapi. The
  API reference is now auto-generated from docstrings; per-module `.rst`
  stubs no longer need to be maintained by hand.
- `docs/` now ships an executable Getting Started notebook (`myst-nb`) which
  re-runs on every build, so the public-API examples cannot rot.
- Intersphinx links to PyTorch / NumPy / SciPy so `:class:torch.nn.*`
  references resolve.

### Fixed

- `PerpLossSSIM.forward` was passing the complex `(x, y)` pair to the
  real-only SSIM conv, raising `RuntimeError` on first use. Now passes the
  precomputed magnitudes (matching the cited perpendicular-loss reference).
- Removed dead branches surfaced by the coverage push: an unreachable
  `elif mask_in_missing:` arm in `BaseMasked._load_from_state_dict`
  (PyTorch's `load_state_dict` hard-codes `strict=True` when calling
  `_load_from_state_dict`, so the precondition is never met), an `if
  weight.is_complex():` check in `MaskedWeightMixin.sparsity` whose two
  branches returned identical values, the real-input fallbacks in
  `transforms._resize_spectrum` (only called with complex spectra from
  `FFTResize`), and the unused `_maybe_bn` helper in `rnn.py`.

## [1.2.0]

### Removed

- The legacy `CVTensor` API and its supporting helpers (`cat`, `roll`,
  `from_polar`, `randn`, and the `torch.Tensor.rect` / `torch.Tensor.polar`
  monkey-patch) have been removed. The package now operates exclusively on
  complex-dtype `torch.Tensor` (typically `torch.cfloat`). Use
  `torch.polar(abs, angle)` and `torch.randn(..., dtype=torch.cfloat)`
  directly.

### Fixed

- Correctness in `SlowLinear` / `SlowConv*` / `SlowConvTranspose*` — the
  Gauss-trick bias was previously off by `b_i * (1 + j)` when `bias=True`.
  `SlowConv*` and `SlowConvTranspose*` now correctly forward `dilation` and
  `output_padding`. The fast (native-cfloat) wrappers were unaffected.
- Complex-valued `BatchNorm*` eval-mode no longer broadcasts `running_mean`
  against the wrong axes.
- `PhaseSigmoid` is now implemented (previously was an empty class).
  `MagMinMaxNorm` now correctly preserves phase (previously it subtracted a
  real scalar from a complex tensor).

### Added

- Fast `ConvTranspose1d` / `ConvTranspose2d` / `ConvTranspose3d` are now
  exported from `complextorch.nn`. Their `output_padding` default matches
  PyTorch's (`0`).
- Complex-valued losses (`CVQuadError`, `CVFourthPowError`, `CVCauchyError`,
  `CVLogCoshError`, `CVLogError`) now accept a `reduction` argument
  (`'mean'` | `'sum'` | `'none'`), defaulting to `'mean'`.
- `complextorch.nn.Conv1d` (and its 2-D / 3-D / transposed siblings) wrap
  `torch.nn.Conv1d` with `dtype=torch.cfloat` for maximum efficiency. The
  hand-rolled real/imag-split convolutions remain available under the
  `Slow` prefix.

[Unreleased]: https://github.com/josiahwsmith10/complextorch/compare/2.0.0...HEAD
[2.0.0]: https://github.com/josiahwsmith10/complextorch/releases/tag/2.0.0
[1.2.0]: https://github.com/josiahwsmith10/complextorch/releases/tag/1.2.0
