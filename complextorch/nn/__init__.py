from complextorch.nn.modules.activation import (
    CVSplitReLU,
    CReLU,
    CPReLU,
    zAbsReLU,
    zLeakyReLU,
    GTReLU,
    EquivariantPhaseReLU,
)
from complextorch.nn.modules.activation import (
    CVSigmoid,
    zReLU,
    CVCardiod,
    CVSigLog,
    Mod,
)
from complextorch.nn.modules.activation import (
    GeneralizedSplitActivation,
    CVSplitTanh,
    CTanh,
    CVSplitSigmoid,
    CSigmoid,
    CVSplitAbs,
    CVSplitELU,
    CELU,
    CVSplitCELU,
    CCELU,
    CVSplitGELU,
    CGELU,
)
from complextorch.nn.modules.activation import (
    GeneralizedPolarActivation,
    CVPolarTanh,
    CVPolarSquash,
    CVPolarLog,
    modReLU,
    AdaptiveModReLU,
)

from complextorch.nn.modules.casting import (
    InterleavedToComplex,
    ComplexToInterleaved,
    ConcatenatedToComplex,
    ComplexToConcatenated,
    RealToComplex,
)
from complextorch.nn.modules.phase import PhaseShift, ComplexScaling

from complextorch.nn.modules.phase_modulation import (
    PhaseDivConv1d,
    PhaseDivConv2d,
    PhaseDivConv3d,
    PhaseConjConv1d,
    PhaseConjConv2d,
    PhaseConjConv3d,
)

from complextorch.nn.modules.prototype import PrototypeDistance

from complextorch.nn.modules.conv import Conv1d, Conv2d, Conv3d
from complextorch.nn.modules.conv import (
    ConvTranspose1d,
    ConvTranspose2d,
    ConvTranspose3d,
)

from complextorch.nn.modules.manifold import (
    wFMConv1d,
    wFMConv2d,
    wFMReLU,
    wFMDistanceLinear,
)

from complextorch.nn.modules.dropout import Dropout, Dropout1d, Dropout2d, Dropout3d

from complextorch.nn.modules.linear import Linear, Bilinear

from complextorch.nn.modules.fft import FFTBlock, IFFTBlock

from complextorch.nn.modules.batchnorm import (
    BatchNorm1d,
    BatchNorm2d,
    BatchNorm3d,
    NaiveBatchNorm1d,
    NaiveBatchNorm2d,
    NaiveBatchNorm3d,
    MagBatchNorm1d,
    MagBatchNorm2d,
    MagBatchNorm3d,
)
from complextorch.nn.modules.layernorm import LayerNorm
from complextorch.nn.modules.rmsnorm import RMSNorm
from complextorch.nn.modules.groupnorm import GroupNorm

from complextorch.nn import init
from complextorch.nn import gauss
from complextorch.nn import relevance
from complextorch.nn import masked
from complextorch.nn import utils

from complextorch.nn.modules.softmax import CVSoftMax, MagSoftMax, PhaseSoftMax

from complextorch.nn.modules.mask import ComplexRatioMask, PhaseSigmoid, MagMinMaxNorm

from complextorch.nn.modules.loss import GeneralizedSplitLoss
from complextorch.nn.modules.loss import (
    SSIM,
    SplitSSIM,
    PerpLossSSIM,
    SplitL1,
    SplitMSE,
)
from complextorch.nn.modules.loss import (
    CVQuadError,
    CVFourthPowError,
    CVCauchyError,
    CVLogCoshError,
)
from complextorch.nn.modules.loss import CVLogError, MSELoss

from complextorch.nn.modules.pooling import (
    AdaptiveAvgPool1d,
    AdaptiveAvgPool2d,
    AdaptiveAvgPool3d,
    AvgPool1d,
    AvgPool2d,
    AvgPool3d,
    MagMaxPool1d,
    MagMaxPool2d,
    MagMaxPool3d,
)

from complextorch.nn.modules.upsampling import Upsample, PolarUpsample

from complextorch.nn.modules.rnn import GRUCell, GRU, LSTMCell, LSTM

from complextorch.nn.modules.transformer import (
    TransformerEncoderLayer,
    TransformerEncoder,
    TransformerDecoderLayer,
    TransformerDecoder,
    Transformer,
)

# dependent on the above
from complextorch.nn.modules.attention import (
    MultiheadAttention,
    ScaledDotProductAttention,
)
from complextorch.nn.modules.attention.eca import (
    EfficientChannelAttention1d,
    EfficientChannelAttention2d,
    EfficientChannelAttention3d,
)
from complextorch.nn.modules.attention.mca import (
    MaskedChannelAttention1d,
    MaskedChannelAttention2d,
    MaskedChannelAttention3d,
)

__all__ = [
    # activations
    "CVSplitReLU",
    "CReLU",
    "CPReLU",
    "zAbsReLU",
    "zLeakyReLU",
    "GTReLU",
    "EquivariantPhaseReLU",
    "CVSigmoid",
    "zReLU",
    "CVCardiod",
    "CVSigLog",
    "Mod",
    "GeneralizedSplitActivation",
    "CVSplitTanh",
    "CTanh",
    "CVSplitSigmoid",
    "CSigmoid",
    "CVSplitAbs",
    "CVSplitELU",
    "CELU",
    "CVSplitCELU",
    "CCELU",
    "CVSplitGELU",
    "CGELU",
    "GeneralizedPolarActivation",
    "CVPolarTanh",
    "CVPolarSquash",
    "CVPolarLog",
    "modReLU",
    "AdaptiveModReLU",
    # casting / phase
    "InterleavedToComplex",
    "ComplexToInterleaved",
    "ConcatenatedToComplex",
    "ComplexToConcatenated",
    "RealToComplex",
    "PhaseShift",
    "ComplexScaling",
    # phase modulation
    "PhaseDivConv1d",
    "PhaseDivConv2d",
    "PhaseDivConv3d",
    "PhaseConjConv1d",
    "PhaseConjConv2d",
    "PhaseConjConv3d",
    # prototype classifier
    "PrototypeDistance",
    # conv
    "Conv1d",
    "Conv2d",
    "Conv3d",
    "ConvTranspose1d",
    "ConvTranspose2d",
    "ConvTranspose3d",
    # manifold
    "wFMConv1d",
    "wFMConv2d",
    "wFMReLU",
    "wFMDistanceLinear",
    # dropout
    "Dropout",
    "Dropout1d",
    "Dropout2d",
    "Dropout3d",
    # linear
    "Linear",
    "Bilinear",
    # fft
    "FFTBlock",
    "IFFTBlock",
    # normalization
    "BatchNorm1d",
    "BatchNorm2d",
    "BatchNorm3d",
    "NaiveBatchNorm1d",
    "NaiveBatchNorm2d",
    "NaiveBatchNorm3d",
    "MagBatchNorm1d",
    "MagBatchNorm2d",
    "MagBatchNorm3d",
    "LayerNorm",
    "RMSNorm",
    "GroupNorm",
    # subpackages
    "init",
    "gauss",
    "relevance",
    "masked",
    "utils",
    # softmax / mask
    "CVSoftMax",
    "MagSoftMax",
    "PhaseSoftMax",
    "ComplexRatioMask",
    "PhaseSigmoid",
    "MagMinMaxNorm",
    # losses
    "GeneralizedSplitLoss",
    "SSIM",
    "SplitSSIM",
    "PerpLossSSIM",
    "SplitL1",
    "SplitMSE",
    "CVQuadError",
    "CVFourthPowError",
    "CVCauchyError",
    "CVLogCoshError",
    "CVLogError",
    "MSELoss",
    # pooling
    "AdaptiveAvgPool1d",
    "AdaptiveAvgPool2d",
    "AdaptiveAvgPool3d",
    "AvgPool1d",
    "AvgPool2d",
    "AvgPool3d",
    "MagMaxPool1d",
    "MagMaxPool2d",
    "MagMaxPool3d",
    # upsampling
    "Upsample",
    "PolarUpsample",
    # rnn
    "GRUCell",
    "GRU",
    "LSTMCell",
    "LSTM",
    # transformer
    "TransformerEncoderLayer",
    "TransformerEncoder",
    "TransformerDecoderLayer",
    "TransformerDecoder",
    "Transformer",
    # attention
    "MultiheadAttention",
    "ScaledDotProductAttention",
    "EfficientChannelAttention1d",
    "EfficientChannelAttention2d",
    "EfficientChannelAttention3d",
    "MaskedChannelAttention1d",
    "MaskedChannelAttention2d",
    "MaskedChannelAttention3d",
]
