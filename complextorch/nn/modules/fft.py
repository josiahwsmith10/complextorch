import torch.nn as nn

from ... import CVTensor

__all__ = ["FFTBlock", "IFFTBlock"]


class FFTBlock(nn.Module):
    r"""
    FFT Block
    ---------

    A complex-valued module that performs the forward fast Fourier transform.

    For more information, see `PyTorch fft <https://pytorch.org/docs/stable/fft.html>`.
    """

    def __init__(self, n=None, dim=-1, norm=None) -> None:
        super(FFTBlock, self).__init__()

        self.n = n
        self.dim = dim
        self.norm = norm

    def forward(self, input: CVTensor) -> CVTensor:
        r"""Performs forward fast Fourier transform (FFT) on the input tensor.

        Args:
            input (CVTensor): input tensor

        Returns:
            CVTensor: fft(x) using configuration established on initialization
        """
        return input.fft(input, n=self.n, dim=self.dim, norm=self.norm)


class IFFTBlock(nn.Module):
    r"""
    IFFT Block
    ----------

    A complex-valued module that performs the inverse fast Fourier transform.

    For more information, see `PyTorch fft <https://pytorch.org/docs/stable/fft.html>`.
    """

    def __init__(self, n=None, dim=-1, norm=None) -> None:
        super(IFFTBlock, self).__init__()

        self.n = n
        self.dim = dim
        self.norm = norm

    def forward(self, input: CVTensor) -> CVTensor:
        r"""Performs inverse fast Fourier transform (FFT) on the input tensor.

        Args:
            input (CVTensor): input tensor

        Returns:
            CVTensor: ifft(x) using configuration established on initialization
        """
        return input.ifft(input, n=self.n, dim=self.dim, norm=self.norm)
