import torch

from . import CVTensor

__all__ = ["randn"]


def randn(*size, dtype=torch.complex64, device="cpu", requires_grad=False) -> CVTensor:
    out = torch.randn(*size, dtype=dtype, device=device, requires_grad=requires_grad)
    return CVTensor(out.real, out.imag)
