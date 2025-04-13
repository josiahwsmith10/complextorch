import torch

__all__ = ["randn"]


def randn(
    *size, dtype=torch.cfloat, device="cpu", requires_grad=False
) -> torch.Tensor:
    return torch.randn(*size, dtype=dtype, device=device, requires_grad=requires_grad)
