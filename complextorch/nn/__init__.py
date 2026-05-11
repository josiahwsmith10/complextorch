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
    wFMConvStrict2d,
    wFMReLU,
    wFMDistanceLinear,
    tReLU,
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
    "CCELU",
    "CELU",
    "CGELU",
    "GRU",
    "LSTM",
    "SSIM",
    # pooling
    "AdaptiveAvgPool1d",
    "AdaptiveAvgPool2d",
    "AdaptiveAvgPool3d",
    "AdaptiveModReLU",
    "AvgPool1d",
    "AvgPool2d",
    "AvgPool3d",
    # normalization
    "BatchNorm1d",
    "BatchNorm2d",
    "BatchNorm3d",
    "Bilinear",
    "CPReLU",
    "CReLU",
    "CSigmoid",
    "CTanh",
    "CVCardiod",
    "CVCauchyError",
    "CVFourthPowError",
    "CVLogCoshError",
    "CVLogError",
    "CVPolarLog",
    "CVPolarSquash",
    "CVPolarTanh",
    "CVQuadError",
    "CVSigLog",
    "CVSigmoid",
    # softmax / mask
    "CVSoftMax",
    "CVSplitAbs",
    "CVSplitCELU",
    "CVSplitELU",
    "CVSplitGELU",
    # activations
    "CVSplitReLU",
    "CVSplitSigmoid",
    "CVSplitTanh",
    "ComplexRatioMask",
    "ComplexScaling",
    "ComplexToConcatenated",
    "ComplexToInterleaved",
    "ConcatenatedToComplex",
    # conv
    "Conv1d",
    "Conv2d",
    "Conv3d",
    "ConvTranspose1d",
    "ConvTranspose2d",
    "ConvTranspose3d",
    # dropout
    "Dropout",
    "Dropout1d",
    "Dropout2d",
    "Dropout3d",
    "EfficientChannelAttention1d",
    "EfficientChannelAttention2d",
    "EfficientChannelAttention3d",
    "EquivariantPhaseReLU",
    # fft
    "FFTBlock",
    # rnn
    "GRUCell",
    "GTReLU",
    "GeneralizedPolarActivation",
    "GeneralizedSplitActivation",
    # losses
    "GeneralizedSplitLoss",
    "GroupNorm",
    "IFFTBlock",
    # casting / phase
    "InterleavedToComplex",
    "LSTMCell",
    "LayerNorm",
    # linear
    "Linear",
    "MSELoss",
    "MagBatchNorm1d",
    "MagBatchNorm2d",
    "MagBatchNorm3d",
    "MagMaxPool1d",
    "MagMaxPool2d",
    "MagMaxPool3d",
    "MagMinMaxNorm",
    "MagSoftMax",
    "MaskedChannelAttention1d",
    "MaskedChannelAttention2d",
    "MaskedChannelAttention3d",
    "Mod",
    # attention
    "MultiheadAttention",
    "NaiveBatchNorm1d",
    "NaiveBatchNorm2d",
    "NaiveBatchNorm3d",
    "PerpLossSSIM",
    "PhaseConjConv1d",
    "PhaseConjConv2d",
    "PhaseConjConv3d",
    # phase modulation
    "PhaseDivConv1d",
    "PhaseDivConv2d",
    "PhaseDivConv3d",
    "PhaseShift",
    "PhaseSigmoid",
    "PhaseSoftMax",
    "PolarUpsample",
    # prototype classifier
    "PrototypeDistance",
    "RMSNorm",
    "RealToComplex",
    "ScaledDotProductAttention",
    "SplitL1",
    "SplitMSE",
    "SplitSSIM",
    "Transformer",
    "TransformerDecoder",
    "TransformerDecoderLayer",
    "TransformerEncoder",
    # transformer
    "TransformerEncoderLayer",
    # upsampling
    "Upsample",
    "gauss",
    # subpackages
    "init",
    "masked",
    "modReLU",
    "relevance",
    "tReLU",
    "utils",
    # manifold
    "wFMConv1d",
    "wFMConv2d",
    "wFMConvStrict2d",
    "wFMDistanceLinear",
    "wFMReLU",
    "zAbsReLU",
    "zLeakyReLU",
    "zReLU",
]
