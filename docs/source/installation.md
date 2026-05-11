# Installation

```{important}
**Install PyTorch first.** PyTorch is **not** installed automatically with
`complextorch` — you must install the wheel matching your CUDA / CPU target
from <https://pytorch.org/get-started/locally/> before installing this package.
```

## From PyPI

```sh
pip install complextorch
```

[PyPI project page →](https://pypi.org/project/complextorch/)

## From source

For local development or to track `main`:

```sh
git clone https://github.com/josiahwsmith10/complextorch.git
cd complextorch
pip install -e .
```

## Optional extras

| Extra | Adds | When to use it |
| --- | --- | --- |
| `complextorch[test]` | `pytest`, `pytest-cov`, `pytest-xdist`, `hypothesis` | Running the test suite locally. |
| `complextorch[docs]` | Sphinx + PyData theme + MyST + autoapi + myst-nb + multiversion | Building these docs locally. |
| `complextorch[datasets]` | `h5py` | Future SAR / MRI dataset readers (`PolSFDataset`, `MICCAI2023`, ...). |
| `complextorch[datasets-alos]` | `rasterio` | ALOS-2 CEOS-format reader (needs system-level GDAL). |

Install combinations with the usual pip syntax, e.g.:

```sh
pip install -e .[test,docs]
```

## Verifying the install

```python
import torch
import complextorch as ctorch

print(ctorch.__version__)

x = torch.randn(8, 5, 7, dtype=torch.cfloat)
y = ctorch.nn.Conv1d(5, 16, kernel_size=3)(x)
print(y.shape, y.dtype)   # torch.Size([8, 16, 5])  torch.complex64
```

If that runs without error, you're set. Continue to the
[Getting Started notebook](examples/getting_started.md).
