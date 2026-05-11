r"""
Differentiable Exponential Integral ``Ei``
==========================================

Custom :class:`torch.autograd.Function` whose forward calls
:func:`scipy.special.expi` on CPU and whose backward uses the closed-form
derivative :math:`\frac{d}{dx} \mathrm{Ei}(x) = e^x / x`.

This is used by complex Variational Dropout to compute an exact KL divergence
between a complex-Gaussian posterior and the log-uniform prior — no closed-form
``Ei`` is currently available in pure torch.
"""

import torch

__all__ = ["ExpiFunction", "torch_expi"]


class ExpiFunction(torch.autograd.Function):
    r"""Differentiable port of :func:`scipy.special.expi`.

    Forward goes to CPU + numpy + scipy; backward is analytical
    (:math:`\frac{d}{dx} \mathrm{Ei}(x) = e^x / x`) and stays on the original
    device. Memory transfer overhead is amortised by the fact that ``Ei`` is
    only evaluated on parameter-shaped tensors, not on data-shaped ones.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        import scipy.special

        ctx.save_for_backward(x)
        x_cpu = x.detach().cpu().numpy()
        output = scipy.special.expi(x_cpu)
        return torch.from_numpy(output).to(x.device, dtype=x.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        (x,) = ctx.saved_tensors
        return grad_output * torch.exp(x) / x


torch_expi = ExpiFunction.apply
