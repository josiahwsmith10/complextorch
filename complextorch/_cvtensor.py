import torch
import numpy as np
from copy import deepcopy

from deprecated import deprecated

__all__ = ["CVTensor"]


"""Patches which enable the rest of the code to function without CVTensors and instead with torch.Tensor"""


@property
def rect(self):
    r"""Return the complex tensor in rectangular form."""
    assert self.is_complex(), "Must call .rect on a complex tensor"
    return (self.real, self.imag)


@property
def polar(self):
    r"""Return the complex tensor in polar form."""
    assert self.is_complex(), "Must call .polar on a complex tensor"
    return (self.abs(), self.angle())


torch.Tensor.rect = rect
torch.Tensor.polar = polar


class CVTensor:
    r"""
    Complex-Valued Tensor
    ---------------------

    Lightweight complex-valued Rensor class.
    Built on the `PyTorch Tensor <https://pytorch.org/docs/stable/tensors.html>`_ with many similar properties and methods.
    """

    def __init__(self, r: torch.Tensor, i: torch.Tensor):
        self.real = r
        self.imag = i

    def __copy__(self):
        r"""Shallow: a new instance with references to the real-imag data."""
        return CVTensor(self.real, self.imag)

    def __deepcopy__(self, memo):
        r"""Deep: a new instance with copies of the real-imag data."""
        real = deepcopy(self.real, memo)
        imag = deepcopy(self.imag, memo)
        return CVTensor(real, imag)

    def __getitem__(self, key):
        r"""Index the complex tensor."""
        return CVTensor(self.real[key], self.imag[key])

    def __setitem__(self, key, value):
        r"""Alter the complex tensor at index inplace."""
        if is_complex(value):
            self.real[key], self.imag[key] = value.real, value.imag
        else:
            self.real[key], self.imag[key] = value, value

    def __iter__(self):
        r"""Iterate over the zero-th dimension of the complex tensor."""
        return map(CVTensor, self.real, self.imag)

    def __reversed__(self):
        r"""Reverse the complex tensor along the zero-th dimension."""
        return CVTensor(reversed(self.real), reversed(self.imag))

    @deprecated(version=1.1, reason="not needed when using torch.Tensor")
    def clone(self):
        r"""Clone a complex tensor."""
        return CVTensor(self.real.clone(), self.imag.clone())

    @property
    @deprecated(
        version=1.1,
        reason="not needed when using torch.Tensor (will need to call as function instead of property)",
    )
    def conj(self):
        r"""Conjugate of the complex tensor."""
        return CVTensor(self.real, -self.imag)

    @property
    @deprecated(
        version=1.1,
        reason="not needed when using torch.Tensor (this returns a torch.Tensor)",
    )
    def complex(self):
        r"""Return the complex tensor in complex form."""
        out = self.real + 1j * self.imag
        return out

    @property
    def rect(self):
        r"""Return the complex tensor in rectangular form."""
        return (self.real, self.imag)

    @property
    def polar(self):
        r"""Return the complex tensor in polar form."""
        return (self.abs(), self.angle())

    @deprecated(
        version=1.1,
        reason="not needed when using torch.Tensor replaced by torch.Tensor.conj()",
    )
    def conjugate(self):
        r"""Conjugate of the complex tensor."""
        return self.conj

    @deprecated(version=1.1, reason="not needed when using torch.Tensor")
    def __pos__(self):
        r"""Positive of the complex tensor."""
        return self

    @deprecated(version=1.1, reason="not needed when using torch.Tensor")
    def __neg__(self):
        r"""Negative of the complex tensor."""
        return CVTensor(-self.real, -self.imag)

    @deprecated(version=1.1, reason="not needed when using torch.Tensor")
    def __add__(self, other):
        r"""Addition of two complex tensors."""
        if is_complex(other):
            return CVTensor(self.real + other.real, self.imag + other.imag)
        else:
            return CVTensor(self.real + other, self.imag)

    __radd__ = __add__
    __iadd__ = __add__

    @deprecated(version=1.1, reason="not needed when using torch.Tensor")
    def add_(self, other):
        r"""Addition of two complex tensors inplace."""
        if is_complex(other):
            self.real += other.real
            self.imag += other.imag
        else:
            self.real += other
        return self

    @deprecated(version=1.1, reason="not needed when using torch.Tensor")
    def __sub__(self, other):
        r"""Subtraction of two complex tensors."""
        if is_complex(other):
            return CVTensor(self.real - other.real, self.imag - other.imag)
        else:
            return CVTensor(self.real - other, self.imag)

    @deprecated(version=1.1, reason="not needed when using torch.Tensor")
    def __rsub__(self, other):
        r"""Subtraction of two complex tensors."""
        return -self + other

    __isub__ = __sub__

    @deprecated(version=1.1, reason="not needed when using torch.Tensor")
    def sub_(self, other):
        r"""Subtraction of two complex tensors inplace."""
        if is_complex(other):
            self.real -= other.real
            self.imag -= other.imag
        else:
            self.real -= other
        return self

    @deprecated(
        version=1.1, reason="not needed when using torch.Tensor, which is faster"
    )
    def __mul__(self, other):
        r"""Multiplication of two complex tensors using Gauss' multiplication trick to reduce computational load."""
        if is_complex(other):
            t1 = self.real * other.real
            t2 = self.imag * other.imag
            t3 = (self.real + self.imag) * (other.real + other.imag)
            return CVTensor(t1 - t2, t3 - t2 - t1)
        else:
            return CVTensor(self.real * other, self.imag * other)

    __rmul__ = __mul__
    __imul__ = __mul__

    @deprecated(version=1.1, reason="not needed when using torch.Tensor")
    def mul_(self, other):
        r"""Multiplication of two complex tensors inplace."""
        if is_complex(other):
            t1 = self.real * other.real
            t2 = self.imag * other.imag
            t3 = (self.real + self.imag) * (other.real + other.imag)
            self.real = t1 - t2
            self.imag = t3 - t2 - t1
        else:
            self.real *= other
            self.imag *= other
        return self

    @deprecated(version=1.1, reason="not needed when using torch.Tensor")
    def __truediv__(self, other):
        r"""Elementwise division of two complex tensors."""
        if is_complex(other):
            return self * other.conjugate() / (other.real**2 + other.imag**2)
        else:
            return CVTensor(self.real / other, self.imag / other)

    def __rtruediv__(self, other):
        r"""Element-wise division of something by the complex tensor."""
        return other * self.conjugate() / (self.real**2 + self.imag**2)

    __itruediv__ = __truediv__

    def div_(self, other):
        r"""Elementwise division of two complex tensors inplace."""
        if is_complex(other):
            self *= other.conjugate() / (other.real**2 + other.imag**2)
        else:
            self.real /= other
            self.imag /= other
        return self

    def __matmul__(self, other):
        r"""
        Matrix multiplication of two complex tensors.
        Using Gauss' multiplication trick to reduce computation time.
        r"""
        if is_complex(other):
            t1 = torch.matmul(self.real, other.real)
            t2 = torch.matmul(self.imag, other.imag)
            t3 = torch.matmul(self.real + self.imag, other.real + other.imag)
            return CVTensor(t1 - t2, t3 - t2 - t1)
        else:
            return CVTensor(
                torch.matmul(self.real, other), torch.matmul(self.imag, other)
            )

    def __rmatmul__(self, other):
        r"""Matrix multiplication by a complex tensor from the right."""
        return CVTensor(torch.matmul(other, self.real), torch.matmul(other, self.imag))

    __imatmul__ = __matmul__

    def __abs__(self):
        r"""Absolute value of the complex tensor."""
        return self.complex.abs()

    def abs(self):
        r"""Absolute value of the complex tensor."""
        return self.__abs__()

    def angle(self):
        r"""Angle of the complex tensor."""
        return self.complex.angle()

    @property
    def shape(self):
        r""" "Shape of the complex tensor."""
        return self.real.shape

    def __len__(self):
        r"""Length of the complex tensor."""
        return len(self.real)

    def t(self):
        r"""Transpose of the complex tensor."""
        return CVTensor(self.real.t(), self.imag.t())

    def h(self):
        r"""Hermitian transpose of the complex tensor."""
        return self.conj.t()

    @property
    def T(self):
        r"""Transpose of the complex tensor."""
        return self.t()

    @property
    def H(self):
        r"""Hermitian transpose of the complex tensor."""
        return self.h()

    def flatten(self, start_dim=0, end_dim=-1):
        r"""Flatten the complex tensor."""
        return CVTensor(
            self.real.flatten(start_dim, end_dim), self.imag.flatten(start_dim, end_dim)
        )

    def view(self, *shape):
        r"""View the complex tensor."""
        shape = shape[0] if shape and isinstance(shape[0], tuple) else shape
        return CVTensor(self.real.view(*shape), self.imag.view(*shape))

    def view_as(self, other):
        r"""View the complex tensor as another tensor."""
        return CVTensor(self.real.view_as(other), self.imag.view_as(other))

    def reshape(self, *shape):
        r"""Reshape the complex tensor."""
        shape = shape[0] if shape and isinstance(shape[0], tuple) else shape
        return CVTensor(self.real.reshape(*shape), self.imag.reshape(*shape))

    def size(self, *dim):
        r"""Size of the complex tensor."""
        return self.real.size(*dim)

    def squeeze(self, dim=None):
        r"""Squeeze the complex tensor."""
        if dim:
            return CVTensor(self.real.squeeze(dim), self.imag.squeeze(dim))
        else:
            return CVTensor(self.real.squeeze(), self.imag.squeeze())

    def unsqueeze(self, dim):
        r"""Unsqueeze the complex tensor."""
        return CVTensor(self.real.unsqueeze(dim), self.imag.unsqueeze(dim))

    def item(self):
        r"""Get the scalar value of the complex tensor if it is zero-dim."""
        return self.complex.item()

    @classmethod
    def from_numpy(cls, x: np.ndarray):
        r"""Create a complex tensor from a numpy array."""
        return cls(torch.from_numpy(x.real), torch.from_numpy(x.imag))

    def numpy(self):
        r"""Convert the complex tensor to a numpy array."""
        return self.complex.numpy()

    def __repr__(self):
        r"""Representation of the complex tensor."""
        return f"CVTensor({self.complex})"

    def detach(self):
        r"""Detach the complex tensor from the computation graph."""
        return CVTensor(self.real.detach(), self.imag.detach())

    def requires_grad_(self, requires_grad=True):
        r"""Set the requires_grad attribute of the complex tensor."""
        return CVTensor(
            self.real.requires_grad_(requires_grad),
            self.imag.requires_grad_(requires_grad),
        )

    def cuda(self, device=None, non_blocking=False):
        r"""Move the complex tensor to the GPU."""
        return CVTensor(
            self.real.cuda(device, non_blocking), self.imag.cuda(device, non_blocking)
        )

    def cpu(self):
        r"""Move the complex tensor to the CPU."""
        return CVTensor(self.real.cpu(), self.imag.cpu())

    def to(self, *args, **kwargs):
        r"""Move the complex tensor to the specified device."""
        return CVTensor(self.real.to(*args, **kwargs), self.imag.to(*args, **kwargs))

    @property
    def device(self):
        r"""Device of the complex tensor."""
        return self.real.device

    @property
    def dtype(self):
        r"""Data type of the complex tensor."""
        return self.real.dtype

    def dim(self):
        r"""Dimension of the complex tensor."""
        return self.real.dim()

    def permute(self, *dims):
        r"""Permute the complex tensor."""
        return CVTensor(self.real.permute(*dims), self.imag.permute(*dims))

    def transpose(self, dim0, dim1):
        r"""Transpose the complex tensor."""
        return CVTensor(
            self.real.transpose(dim0, dim1), self.imag.transpose(dim0, dim1)
        )

    def is_complex(self):
        r"""Check if the complex tensor is complex."""
        return True

    def contiguous(self, memory_format=torch.contiguous_format):
        r"""Make the complex tensor contiguous."""
        return CVTensor(
            self.real.contiguous(memory_format=memory_format),
            self.imag.contiguous(memory_format=memory_format),
        )

    def clone(self):
        r"""Clone the complex tensor."""
        return CVTensor(self.real.clone(), self.imag.clone())

    def expand(self, *sizes, **kwargs):
        r"""Expand the complex tensor."""
        return CVTensor(
            self.real.expand(*sizes, **kwargs), self.imag.expand(*sizes, **kwargs)
        )

    def expand_as(self, other):
        r"""Expand the complex tensor as another tensor."""
        if is_complex(other):
            return CVTensor(
                self.real.expand_as(other.real), self.imag.expand_as(other.imag)
            )
        else:
            return CVTensor(self.real.expand_as(other), self.imag.expand_as(other))

    def fft(self, n=None, dim=-1, norm="ortho"):
        r"""FFT of the complex tensor."""
        out = torch.fft.fft(self.complex, n=n, dim=dim, norm=norm)
        return CVTensor(out.real, out.imag)

    def fft_(self, n=None, dim=-1, norm="ortho"):
        r"""In-place FFT of the complex tensor."""
        out = torch.fft.fft(self.complex, n=n, dim=dim, norm=norm)
        self.real, self.imag = out.real, out.imag
        return self

    def ifft(self, n=None, dim=-1, norm="ortho"):
        r"""Inverse FFT of the complex tensor."""
        out = torch.fft.ifft(self.complex, n=n, dim=dim, norm=norm)
        return CVTensor(out.real, out.imag)

    def ifft_(self, n=None, dim=-1, norm="ortho"):
        r"""In-place inverse FFT of the complex tensor."""
        out = torch.fft.ifft(self.complex, n=n, dim=dim, norm=norm)
        self.real, self.imag = out.real, out.imag
        return self

    def mean(self, dim=None, keepdim=False):
        r"""Mean of the complex tensor."""
        return CVTensor(self.real.mean(dim, keepdim), self.imag.mean(dim, keepdim))

    def sum(self, dim=None, keepdim=False):
        r"""Sum of the complex tensor."""
        return CVTensor(self.real.sum(dim, keepdim), self.imag.sum(dim, keepdim))

    def roll(self, shifts, dims=None):
        r"""Same as torch.roll() but for CVTensor."""
        return CVTensor(self.real.roll(shifts, dims), self.imag.roll(shifts, dims))


def cat(tensors, dim=0, out=None) -> CVTensor:
    r"""
    complextorch.cat
    ----------------

    Same as `torch.cat() <https://pytorch.org/docs/stable/generated/torch.cat.html>`_ but for CVTensor.
    r"""
    real = torch.cat([t.real for t in tensors], dim, out=out.real if out else None)
    imag = torch.cat([t.imag for t in tensors], dim, out=out.imag if out else None)
    return CVTensor(real, imag)


def roll(x: torch.Tensor, shifts, dims=None) -> CVTensor:
    r"""
    complextorch.roll
    -----------------

    Same as `torch.roll() <https://pytorch.org/docs/stable/generated/torch.roll.html>`_ but for CVTensor.
    r"""
    return CVTensor(x.real.roll(shifts, dims), x.imag.roll(shifts, dims))


def is_complex(x) -> bool:
    r"""
    is complex
    ----------

    A helper function to determine if an input object comprises complex-valued data either in the form of a CVTensor object or complex-valued `PyTorch Tensor <https://pytorch.org/docs/stable/tensors.html>`_.
    Essentially confirms that the input object has the properties *real* and *imag*.

    Args:
        x: object to test

    Returns:
        bool: Boolean whether or not the input object is complex-valued
    """
    return isinstance(x, (CVTensor, complex)) or (
        torch.is_complex(x) if torch.is_tensor(x) else False
    )


def from_polar(r: torch.Tensor, phi: torch.Tensor) -> CVTensor:
    r"""
    from polar
    ----------

    Create a CVTensor from polar formatted tensors containing the magnitude (:math:`r`) and phase (:math:`\phi`) information.

    Implements the following operation:

    .. math::

        G(r, \phi) = r * \exp(j * \phi)
    """
    return torch.complex(r * torch.cos(phi), r * torch.sin(phi))
