import torch.nn as nn

from ... import CVTensor

__all__ = ["FFTBlock", "IFFTBlock"]


class FFTBlock(nn.Module):
    """
    FFT Block
    ---------
    
    A complex-valued module that performs the forward fast Fourier transform. 
    """

    def __init__(self, n=None, dim=-1, norm=None) -> None:
        super(FFTBlock, self).__init__()

        self.n = n
        self.dim = dim
        self.norm = norm

    def forward(self, x: CVTensor) -> CVTensor:
        """Performs forward fast Fourier transform (FFT) on the input tensor.

        Args:
            x (CVTensor): input tensor

        Returns:
            CVTensor: fft(x) using configuration established on initialization
        """
        return x.fft(x, n=self.n, dim=self.dim, norm=self.norm)


class IFFTBlock(nn.Module):
    """
    IFFT Block
    ----------
    
    A complex-valued module that performs the inverse fast Fourier transform. 
    """

    def __init__(self, n=None, dim=-1, norm=None) -> None:
        super(IFFTBlock, self).__init__()

        self.n = n
        self.dim = dim
        self.norm = norm

    def forward(self, x: CVTensor) -> CVTensor:
        """Performs inverse fast Fourier transform (FFT) on the input tensor.

        Args:
            x (CVTensor): input tensor

        Returns:
            CVTensor: ifft(x) using configuration established on initialization
        """
        return x.ifft(x, n=self.n, dim=self.dim, norm=self.norm)
