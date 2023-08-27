import torch.nn as nn

from ... import CVTensor

__all__ = ["FFTBlock", "IFFTBlock"]


class FFTBlock(nn.Module):
    """FFT Block."""

    def __init__(self, n=None, dim=-1, norm=None) -> None:
        super(FFTBlock, self).__init__()

        self.n = n
        self.dim = dim
        self.norm = norm

    def forward(self, x: CVTensor) -> CVTensor:
        return x.fft(x, n=self.n, dim=self.dim, norm=self.norm)


class IFFTBlock(nn.Module):
    """IFFT Block."""

    def __init__(self, n=None, dim=-1, norm=None) -> None:
        super(IFFTBlock, self).__init__()

        self.n = n
        self.dim = dim
        self.norm = norm

    def forward(self, x: CVTensor) -> CVTensor:
        return x.ifft(x, n=self.n, dim=self.dim, norm=self.norm)
