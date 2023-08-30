import torch
import numpy as np
from copy import deepcopy

__all__ = ["CVTensor"]


class CVTensor:
    """
    Complex-Valued Tensor
    ---------------------

    Lightweight complex-valued Rensor class.
    Built on the `PyTorch Tensor <https://pytorch.org/docs/stable/tensors.html>`_ with many similar properties and methods.
    """

    def __init__(self, r: torch.Tensor, i: torch.Tensor):
        self.real = r
        self.imag = i

    def __copy__(self):
        """Shallow: a new instance with references to the real-imag data."""
        return CVTensor(self.real, self.imag)

    def __deepcopy__(self, memo):
        """Deep: a new instance with copies of the real-imag data."""
        real = deepcopy(self.real, memo)
        imag = deepcopy(self.imag, memo)
        return CVTensor(real, imag)

    def __getitem__(self, key):
        """Index the complex tensor."""
        return CVTensor(self.real[key], self.imag[key])

    def __setitem__(self, key, value):
        """Alter the complex tensor at index inplace."""
        if is_complex(value):
            self.real[key], self.imag[key] = value.real, value.imag
        else:
            self.real[key], self.imag[key] = value, value

    def __iter__(self):
        """Iterate over the zero-th dimension of the complex tensor."""
        return map(CVTensor, self.real, self.imag)

    def __reversed__(self):
        """Reverse the complex tensor along the zero-th dimension."""
        return CVTensor(reversed(self.real), reversed(self.imag))

    def clone(self):
        """Clone a complex tensor."""
        return CVTensor(self.real.clone(), self.imag.clone())

    @property
    def conj(self):
        """Conjugate of the complex tensor."""
        return CVTensor(self.real, -self.imag)

    @property
    def complex(self):
        """Return the complex tensor in complex form."""
        out = self.real + 1j * self.imag
        return out

    @property
    def rect(self):
        """Return the complex tensor in rectangular form."""
        return (self.real, self.imag)

    @property
    def polar(self):
        """Return the complex tensor in polar form."""
        return (self.abs(), self.angle())

    def conjugate(self):
        """Conjugate of the complex tensor."""
        return self.conj

    def __pos__(self):
        """Positive of the complex tensor."""
        return self

    def __neg__(self):
        """Negative of the complex tensor."""
        return CVTensor(-self.real, -self.imag)

    def __add__(self, other):
        """Addition of two complex tensors."""
        if is_complex(other):
            return CVTensor(self.real + other.real, self.imag + other.imag)
        else:
            return CVTensor(self.real + other, self.imag)

    __radd__ = __add__
    __iadd__ = __add__

    def add_(self, other):
        """Addition of two complex tensors inplace."""
        if is_complex(other):
            self.real += other.real
            self.imag += other.imag
        else:
            self.real += other
        return self

    def __sub__(self, other):
        """Subtraction of two complex tensors."""
        if is_complex(other):
            return CVTensor(self.real - other.real, self.imag - other.imag)
        else:
            return CVTensor(self.real - other, self.imag)

    def __rsub__(self, other):
        """Subtraction of two complex tensors."""
        return -self + other

    __isub__ = __sub__

    def sub_(self, other):
        """Subtraction of two complex tensors inplace."""
        if is_complex(other):
            self.real -= other.real
            self.imag -= other.imag
        else:
            self.real -= other
        return self

    def __mul__(self, other):
        """Multiplication of two complex tensors using Gauss' multiplication trick to reduce computational load."""
        if is_complex(other):
            t1 = self.real * other.real
            t2 = self.imag * other.imag
            t3 = (self.real + self.imag) * (other.real + other.imag)
            return CVTensor(t1 - t2, t3 - t2 - t1)
        else:
            return CVTensor(self.real * other, self.imag * other)

    __rmul__ = __mul__
    __imul__ = __mul__

    def mul_(self, other):
        """Multiplication of two complex tensors inplace."""
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

    def __truediv__(self, other):
        """Elementwise division of two complex tensors."""
        if is_complex(other):
            return self * other.conjugate() / (other.real**2 + other.imag**2)
        else:
            return CVTensor(self.real / other, self.imag / other)

    def __rtruediv__(self, other):
        """Element-wise division of something by the complex tensor."""
        return other * self.conjugate() / (self.real**2 + self.imag**2)

    __itruediv__ = __truediv__

    def div_(self, other):
        """Elementwise division of two complex tensors inplace."""
        if is_complex(other):
            self *= other.conjugate() / (other.real**2 + other.imag**2)
        else:
            self.real /= other
            self.imag /= other
        return self

    def __matmul__(self, other):
        """
        Matrix multiplication of two complex tensors.
        Using Gauss' multiplication trick to reduce computation time.
        """
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
        """Matrix multiplication by a complex tensor from the right."""
        return CVTensor(torch.matmul(other, self.real), torch.matmul(other, self.imag))

    __imatmul__ = __matmul__

    def __abs__(self):
        """Absolute value of the complex tensor."""
        return self.complex.abs()

    def abs(self):
        """Absolute value of the complex tensor."""
        return self.__abs__()

    def angle(self):
        """Angle of the complex tensor."""
        return self.complex.angle()

    @property
    def shape(self):
        """ "Shape of the complex tensor."""
        return self.real.shape

    def __len__(self):
        """Length of the complex tensor."""
        return len(self.real)

    def t(self):
        """Transpose of the complex tensor."""
        return CVTensor(self.real.t(), self.imag.t())

    def h(self):
        """Hermitian transpose of the complex tensor."""
        return self.conj.t()

    @property
    def T(self):
        """Transpose of the complex tensor."""
        return self.t()

    @property
    def H(self):
        """Hermitian transpose of the complex tensor."""
        return self.h()

    def flatten(self, start_dim=0, end_dim=-1):
        """Flatten the complex tensor."""
        return CVTensor(
            self.real.flatten(start_dim, end_dim), self.imag.flatten(start_dim, end_dim)
        )

    def view(self, *shape):
        """View the complex tensor."""
        shape = shape[0] if shape and isinstance(shape[0], tuple) else shape
        return CVTensor(self.real.view(*shape), self.imag.view(*shape))

    def view_as(self, other):
        """View the complex tensor as another tensor."""
        return CVTensor(self.real.view_as(other), self.imag.view_as(other))

    def reshape(self, *shape):
        """Reshape the complex tensor."""
        shape = shape[0] if shape and isinstance(shape[0], tuple) else shape
        return CVTensor(self.real.reshape(*shape), self.imag.reshape(*shape))

    def size(self, *dim):
        """Size of the complex tensor."""
        return self.real.size(*dim)

    def squeeze(self, dim=None):
        """Squeeze the complex tensor."""
        if dim:
            return CVTensor(self.real.squeeze(dim), self.imag.squeeze(dim))
        else:
            return CVTensor(self.real.squeeze(), self.imag.squeeze())

    def unsqueeze(self, dim):
        """Unsqueeze the complex tensor."""
        return CVTensor(self.real.unsqueeze(dim), self.imag.unsqueeze(dim))

    def item(self):
        """Get the scalar value of the complex tensor if it is zero-dim."""
        return self.complex.item()

    @classmethod
    def from_numpy(cls, x: np.ndarray):
        """Create a complex tensor from a numpy array."""
        return cls(torch.from_numpy(x.real), torch.from_numpy(x.imag))

    def numpy(self):
        """Convert the complex tensor to a numpy array."""
        return self.complex.numpy()

    def __repr__(self):
        """Representation of the complex tensor."""
        return f"CVTensor({self.complex})"

    def detach(self):
        """Detach the complex tensor from the computation graph."""
        return CVTensor(self.real.detach(), self.imag.detach())

    def requires_grad_(self, requires_grad=True):
        """Set the requires_grad attribute of the complex tensor."""
        return CVTensor(
            self.real.requires_grad_(requires_grad),
            self.imag.requires_grad_(requires_grad),
        )

    def cuda(self, device=None, non_blocking=False):
        """Move the complex tensor to the GPU."""
        return CVTensor(
            self.real.cuda(device, non_blocking), self.imag.cuda(device, non_blocking)
        )

    def cpu(self):
        """Move the complex tensor to the CPU."""
        return CVTensor(self.real.cpu(), self.imag.cpu())

    def to(self, *args, **kwargs):
        """Move the complex tensor to the specified device."""
        return CVTensor(self.real.to(*args, **kwargs), self.imag.to(*args, **kwargs))

    @property
    def device(self):
        """Device of the complex tensor."""
        return self.real.device

    @property
    def dtype(self):
        """Data type of the complex tensor."""
        return self.real.dtype

    def dim(self):
        """Dimension of the complex tensor."""
        return self.real.dim()

    def permute(self, *dims):
        """Permute the complex tensor."""
        return CVTensor(self.real.permute(*dims), self.imag.permute(*dims))

    def transpose(self, dim0, dim1):
        """Transpose the complex tensor."""
        return CVTensor(
            self.real.transpose(dim0, dim1), self.imag.transpose(dim0, dim1)
        )

    def is_complex(self):
        """Check if the complex tensor is complex."""
        return True

    def contiguous(self, memory_format=torch.contiguous_format):
        """Make the complex tensor contiguous."""
        return CVTensor(
            self.real.contiguous(memory_format=memory_format),
            self.imag.contiguous(memory_format=memory_format),
        )

    def clone(self):
        """Clone the complex tensor."""
        return CVTensor(self.real.clone(), self.imag.clone())

    def expand(self, *sizes, **kwargs):
        """Expand the complex tensor."""
        return CVTensor(
            self.real.expand(*sizes, **kwargs), self.imag.expand(*sizes, **kwargs)
        )

    def expand_as(self, other):
        """Expand the complex tensor as another tensor."""
        if is_complex(other):
            return CVTensor(
                self.real.expand_as(other.real), self.imag.expand_as(other.imag)
            )
        else:
            return CVTensor(self.real.expand_as(other), self.imag.expand_as(other))

    def fft(self, n=None, dim=-1, norm="ortho"):
        """FFT of the complex tensor."""
        out = torch.fft.fft(self.complex, n=n, dim=dim, norm=norm)
        return CVTensor(out.real, out.imag)

    def fft_(self, n=None, dim=-1, norm="ortho"):
        """In-place FFT of the complex tensor."""
        out = torch.fft.fft(self.complex, n=n, dim=dim, norm=norm)
        self.real, self.imag = out.real, out.imag
        return self

    def ifft(self, n=None, dim=-1, norm="ortho"):
        """Inverse FFT of the complex tensor."""
        out = torch.fft.ifft(self.complex, n=n, dim=dim, norm=norm)
        return CVTensor(out.real, out.imag)

    def ifft_(self, n=None, dim=-1, norm="ortho"):
        """In-place inverse FFT of the complex tensor."""
        out = torch.fft.ifft(self.complex, n=n, dim=dim, norm=norm)
        self.real, self.imag = out.real, out.imag
        return self

    def mean(self, dim=None, keepdim=False):
        """Mean of the complex tensor."""
        return CVTensor(self.real.mean(dim, keepdim), self.imag.mean(dim, keepdim))

    def sum(self, dim=None, keepdim=False):
        """Sum of the complex tensor."""
        return CVTensor(self.real.sum(dim, keepdim), self.imag.sum(dim, keepdim))

    def roll(self, shifts, dims=None):
        """Same as torch.roll() but for CVTensor."""
        return CVTensor(self.real.roll(shifts, dims), self.imag.roll(shifts, dims))


def cat(tensors, dim=0, out=None) -> CVTensor:
    """
    complextorch.cat
    ----------------

    Same as `torch.cat() <https://pytorch.org/docs/stable/generated/torch.cat.html>`_ but for CVTensor.
    """
    real = torch.cat([t.real for t in tensors], dim, out=out.real if out else None)
    imag = torch.cat([t.imag for t in tensors], dim, out=out.imag if out else None)
    return CVTensor(real, imag)


def roll(x: torch.Tensor, shifts, dims=None) -> CVTensor:
    """
    complextorch.roll
    -----------------

    Same as `torch.roll() <https://pytorch.org/docs/stable/generated/torch.roll.html>`_ but for CVTensor.
    """
    return CVTensor(x.real.roll(shifts, dims), x.imag.roll(shifts, dims))


def is_complex(x) -> bool:
    """
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
    """
    from polar
    ----------

    Create a CVTensor from polar formatted tensors containing the magnitude (:math:`r`) and phase (:math:`\phi`) information.

    Implements the following operation:

    .. math::

        G(r, \phi) = r * \exp(j * \phi)
    """
    return CVTensor(r * torch.cos(phi), r * torch.sin(phi))
