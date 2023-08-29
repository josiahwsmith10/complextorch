import torch.nn as nn

from ... import CVTensor

__all__ = ["ComplexRatioMask"]


class ComplexRatioMask(nn.Module):
    """
    Complex Ratio Mask
    ------------------
    
    .. math::

        G(\mathbf{z}) = \\text{sigmoid}(|\mathbf{z}|) * \mathbf{z} / |\mathbf{z}|

    Retains phase and squeezes magnitude using `sigmoid function <https://pytorch.org/docs/stable/generated/torch.nn.Sigmoid.html>`_.

    Based on work from the following paper:

        **HW Cho, S Choi, YR Cho, and J Kim: Complex-Valued Channel Attention and Application in Ego-Velocity Estimation With Automotive Radar**

            - See [23]

            - https://ieeexplore.ieee.org/abstract/document/9335579
    """

    def __init__(self) -> None:
        super(ComplexRatioMask, self).__init__()

    def forward(self, input: CVTensor) -> CVTensor:
        """Computes complex ratio mask on complex-valued input tensor.

        Args:
            input (CVTensor): input tensor

        Returns:
            CVTensor: :math:`\\text{sigmoid}(|\mathbf{z}|) * \mathbf{z} / |\mathbf{z}|`
        """
        x_mag = input.abs()
        return x_mag.sigmoid() * (input / x_mag)
