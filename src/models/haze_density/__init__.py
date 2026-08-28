"""
雾密度感知网络模块

根据《最终版申报书.pdf》3.2.1 节实现。

网络架构:
    Input (B, 3, H, W)
        ↓
    Encoder (下采样)
        ↓
    3 路并行多尺度 SDRB (dilation=2,3,4)
        → 每路：SDRB → RB → RB → ECA
        ↓
    Concat + 3×3 Conv + ECA
        ↓
    Decoder (上采样)
        ↓
    Output (B, 1, H, W)

子模块:
    - physical_prior: 物理先验雾密度估计
    - guided_filter: 引导滤波
    - basic_blocks: 基础卷积块
    - residual_blocks: 残差块 (RB/SDRB)
    - eca: ECA 通道注意力
    - encoder: 编码器
    - decoder: 解码器
    - haze_density_net: 完整网络
"""

# 已实现模块导入
from .physical_prior import (
    dark_channel,
    local_contrast,
    color_shift,
    min_max_normalize,
    compute_physical_prior,
    generate_s_final,
    PhysicalPriorModule,
    WEIGHT_DARK,
    WEIGHT_CONTRAST,
    WEIGHT_COLOR,
    EXPONENT_MU,
)
from .guided_filter import guided_filter, GuidedFilter
from .basic_blocks import ConvBlock, DownsampleBlock, DoubleConvBlock
from .encoder import Encoder
from .residual_blocks import ResidualBlock, DilatedResidualBlock, SDRB
from .eca import ECA, ECAv2
from .multiscale import (
    MultiScaleBranch,
    MultiScaleFeatureExtractor,
    ParallelMultiScaleFeatureExtractor,
)

__all__ = [
    # Physical Prior
    "dark_channel",
    "local_contrast",
    "color_shift",
    "min_max_normalize",
    "compute_physical_prior",
    "generate_s_final",
    "PhysicalPriorModule",
    # Constants
    "WEIGHT_DARK",
    "WEIGHT_CONTRAST",
    "WEIGHT_COLOR",
    "EXPONENT_MU",
    # Guided Filter
    "guided_filter",
    "GuidedFilter",
    # Basic Blocks
    "ConvBlock",
    "DownsampleBlock",
    "DoubleConvBlock",
    # Encoder
    "Encoder",
    # Residual Blocks
    "ResidualBlock",
    "DilatedResidualBlock",
    "SDRB",
    # ECA
    "ECA",
    "ECAv2",
    # MultiScale
    "MultiScaleBranch",
    "MultiScaleFeatureExtractor",
    "ParallelMultiScaleFeatureExtractor",
]
