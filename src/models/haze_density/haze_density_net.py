# -*- coding: utf-8 -*-
"""
雾密度感知网络完整模型

根据《最终版申报书.pdf》3.2.1 节实现。

网络架构:
    Input RGB [B, 3, H, W]
        ↓
    Encoder (下采样)
        ↓ F0 [B, 64, H/2, W/2]
    MultiScale (3 路并行多尺度 SDRB)
        → Branch 1 (dilation=2): SDRB → RB → RB → ECA
        → Branch 2 (dilation=3): SDRB → RB → RB → ECA
        → Branch 3 (dilation=4): SDRB → RB → RB → ECA
        ↓
    Fusion (Concat + Conv + ECA)
        ↓ F_fuse [B, 64, H/2, W/2]
    Decoder (上采样)
        ↓
    Output I_h [B, 1, H, W]

输入:
    image: 含雾 RGB 图像 [B, 3, H, W], 范围 [0, 1]

输出:
    I_h: 雾密度图 [B, 1, H, W], 范围 [0, 1]

注意:
    physical prior 不作为模型 forward 的一部分，
    因为 S_final 是监督信号而不是网络输入特征。

训练时:
    pred = model(image)
    target = physical_prior(image)  # 在外部计算
    loss = criterion(pred, target)

工程实现参数:
    - base_channels: 基础通道数（默认 32）
    - use_sigmoid: Decoder 是否使用 Sigmoid（默认 True）
"""

import torch
import torch.nn as nn

from .encoder import Encoder
from .multiscale import ParallelMultiScaleFeatureExtractor
from .fusion import FusionModule
from .decoder import Decoder


class HazeDensityNet(nn.Module):
    """
    雾密度感知网络

    完整网络结构:
        Encoder → MultiScale → Fusion → Decoder

    Args:
        base_channels: 基础通道数（工程实现参数，默认 32）
        use_sigmoid: Decoder 是否使用 Sigmoid（工程实现决策，默认 True）
    """

    def __init__(
        self,
        base_channels: int = 32,
        use_sigmoid: bool = True,
    ):
        super().__init__()

        self.base_channels = base_channels
        self.use_sigmoid = use_sigmoid

        # Encoder: [B, 3, H, W] -> [B, base_channels*2, H/2, W/2]
        self.encoder = Encoder(base_channels=base_channels)

        # MultiScale: [B, C, H/2, W/2] -> [B, 3*C, H/2, W/2]
        # 其中 C = base_channels * 2
        self.multiscale = ParallelMultiScaleFeatureExtractor(
            channels=base_channels * 2,
            dilation_rates=[2, 3, 4],  # 申报书规定参数
        )

        # Fusion: [B, 3*C, H/2, W/2] -> [B, C, H/2, W/2]
        self.fusion = FusionModule(
            in_channels=base_channels * 2,
            out_channels=base_channels * 2,
        )

        # Decoder: [B, C, H/2, W/2] -> [B, 1, H, W]
        self.decoder = Decoder(
            in_channels=base_channels * 2,
            use_sigmoid=use_sigmoid,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入图像 [B, 3, H, W]

        Returns:
            I_h: 雾密度图 [B, 1, H, W]
        """
        # Encoder
        f0 = self.encoder(x)  # [B, 64, H/2, W/2]

        # MultiScale (返回三个分支的输出)
        f1, f2, f3, _ = self.multiscale(f0, return_separate=True)  # 每个 [B, 64, H/2, W/2]

        # Fusion
        f_fuse = self.fusion(f1, f2, f3)  # [B, 64, H/2, W/2]

        # Decoder
        i_h = self.decoder(f_fuse)  # [B, 1, H, W]

        return i_h

    def extra_repr(self) -> str:
        return (
            f"base_channels={self.base_channels}, "
            f"use_sigmoid={self.use_sigmoid}"
        )

    def count_parameters(self) -> int:
        """计算参数量"""
        return sum(p.numel() for p in self.parameters())

    def get_parameter_stats(self) -> dict:
        """获取各模块参数量统计"""
        stats = {
            "total": self.count_parameters(),
            "encoder": self.encoder.count_parameters(),
            "multiscale": self.multiscale.count_parameters(),
            "fusion": self.fusion.count_parameters(),
            "decoder": self.decoder.count_parameters(),
        }
        return stats


def get_model_summary(model: HazeDensityNet) -> str:
    """
    获取模型摘要信息

    Args:
        model: HazeDensityNet 实例

    Returns:
        摘要字符串
    """
    stats = model.get_parameter_stats()

    summary = []
    summary.append("=" * 60)
    summary.append("HazeDensityNet Model Summary")
    summary.append("=" * 60)
    summary.append(f"base_channels: {model.base_channels}")
    summary.append(f"use_sigmoid: {model.use_sigmoid}")
    summary.append("")
    summary.append("Parameter Statistics:")
    summary.append(f"  Encoder:    {stats['encoder']:,} ({stats['encoder']/stats['total']*100:.1f}%)")
    summary.append(f"  MultiScale: {stats['multiscale']:,} ({stats['multiscale']/stats['total']*100:.1f}%)")
    summary.append(f"  Fusion:     {stats['fusion']:,} ({stats['fusion']/stats['total']*100:.1f}%)")
    summary.append(f"  Decoder:    {stats['decoder']:,} ({stats['decoder']/stats['total']*100:.1f}%)")
    summary.append("-" * 60)
    summary.append(f"  Total:      {stats['total']:,}")
    summary.append("=" * 60)

    return "\n".join(summary)
