# -*- coding: utf-8 -*-
"""
Density Guidance Module (Stage 6-3A)

使用冻结的 HazeDensityNet 输出的 density map，对视觉 backbone feature 进行空间引导。

核心逻辑:
    Density Map [B,1,H0,W0]
        ↓
    Downsample to Feature Resolution (bilinear interpolation)
        ↓
    Density Projection (1x1 Conv: 1 → C)
        ↓
    + Visual Projection (1x1 Conv: C → C)
        ↓
    Sigmoid → Attention Map [B,C,H,W]
        ↓
    Guided Feature = Feature + gamma * Feature * Attention

设计要点:
1. 输入输出 shape 完全一致
2. gamma=0 初始化，保证初始时 output ≈ input (identity)
3. 密度图使用 bilinear interpolation 对齐 feature 分辨率
4. 残差连接保证信息不丢失

接口:
    guided_feature = module(visual_feature, density_map)

    visual_feature: [B, C, H, W]
    density_map:    [B, 1, H0, W0] (可以是任意分辨率)
    guided_feature: [B, C, H, W] (与 visual_feature 相同)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class DensityGuidanceModule(nn.Module):
    """
    密度图引导注意力模块

    Args:
        feature_channels: 视觉特征通道数 C
        density_channels: 密度图通道数 (默认 1)
        use_bias: 是否使用 bias (默认 False)

    结构:
        Density Path:
            density_map [B,1,H0,W0]
                ↓
            F.interpolate (bilinear) → [B,1,H,W]
                ↓
            density_proj (1x1 Conv: 1→C) → [B,C,H,W]

        Visual Path:
            visual_feature [B,C,H,W]
                ↓
            visual_proj (1x1 Conv: C→C) → [B,C,H,W]

        Fusion:
            density_proj + visual_proj
                ↓
            sigmoid → attention [B,C,H,W]

        Output:
            visual_feature + gamma * visual_feature * attention
    """

    def __init__(
        self,
        feature_channels: int,
        density_channels: int = 1,
        use_bias: bool = False,
    ):
        super().__init__()

        self.feature_channels = feature_channels
        self.density_channels = density_channels

        # Density Projection: 1x1 Conv, 1 channel → C channels
        self.density_proj = nn.Conv2d(
            in_channels=density_channels,
            out_channels=feature_channels,
            kernel_size=1,
            bias=use_bias,
        )

        # Visual Projection: 1x1 Conv, C channels → C channels
        self.visual_proj = nn.Conv2d(
            in_channels=feature_channels,
            out_channels=feature_channels,
            kernel_size=1,
            bias=use_bias,
        )

        # Gamma: 可学习缩放因子
        # 初始化为 0，保证初始时 output ≈ input (identity initialization)
        # 这样在训练初期不会破坏已有的 backbone 表征
        self.gamma = nn.Parameter(torch.zeros(1))

        # Sigmoid activation
        self.sigmoid = nn.Sigmoid()

    def forward(
        self,
        visual_feature: torch.Tensor,
        density_map: torch.Tensor,
    ) -> torch.Tensor:
        """
        前向传播

        Args:
            visual_feature: 视觉特征 [B, C, H, W]
            density_map: 密度图 [B, 1, H0, W0] (可以是任意分辨率)

        Returns:
            guided_feature: 引导后的特征 [B, C, H, W]
        """
        # 获取目标分辨率 (与 visual_feature 一致)
        target_size = visual_feature.shape[-2:]  # (H, W)

        # Density Map 对齐：bilinear interpolation
        # 从 [B, 1, H0, W0] 下采样到 [B, 1, H, W]
        if density_map.shape[-2:] != target_size:
            density_aligned = F.interpolate(
                density_map,
                size=target_size,
                mode='bilinear',
                align_corners=False,
            )
        else:
            density_aligned = density_map

        # Density Projection: [B, 1, H, W] → [B, C, H, W]
        density_proj = self.density_proj(density_aligned)

        # Visual Projection: [B, C, H, W] → [B, C, H, W]
        visual_proj = self.visual_proj(visual_feature)

        # Fusion: element-wise addition
        fusion = density_proj + visual_proj

        # Attention Map: sigmoid → [0, 1]
        attention = self.sigmoid(fusion)

        # Guided Feature: residual connection with scaling
        # output = visual_feature + gamma * visual_feature * attention
        # 当 gamma=0 时，output = visual_feature (identity)
        guided_feature = visual_feature + self.gamma * visual_feature * attention

        return guided_feature

    def count_parameters(self) -> int:
        """计算参数量"""
        return sum(p.numel() for p in self.parameters())

    def extra_repr(self) -> str:
        return (
            f"feature_channels={self.feature_channels}, "
            f"density_channels={self.density_channels}"
        )


def create_density_guidance_modules(
    feature_channels_list: Tuple[int, ...] = (128, 256, 512, 1024),
) -> nn.ModuleList:
    """
    为多尺度特征创建 DensityGuidanceModule 列表

    Args:
        feature_channels_list: 各尺度特征通道数列表

    Returns:
        nn.ModuleList of DensityGuidanceModule
    """
    modules = nn.ModuleList([
        DensityGuidanceModule(feature_channels=c)
        for c in feature_channels_list
    ])
    return modules


__all__ = ["DensityGuidanceModule", "create_density_guidance_modules"]
