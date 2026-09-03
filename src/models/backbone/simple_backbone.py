# -*- coding: utf-8 -*-
"""
Simple Backbone for Baseline Testing (Stage 6-2)

用于 Stage 6-2 Baseline Integration 的简化 Backbone 占位模型。

设计目标:
1. 支持 3 通道 (RGB) 和 4 通道 (RGB+Density) 输入
2. 提供多尺度特征输出
3. 易于扩展为完整模型

结构:
    Input [B, C, H, W] (C=3 or 4)
        ↓
    Stem (7x7 conv, stride=2)
        ↓ [B, 64, H/2, W/2]
    Stage 1 (3x3 conv, stride=1)
        ↓ [B, 128, H/4, W/4]
    Stage 2 (3x3 conv, stride=2)
        ↓ [B, 256, H/8, W/8]
    Stage 3 (3x3 conv, stride=2)
        ↓ [B, 512, H/16, W/16]
    Stage 4 (3x3 conv, stride=2)
        ↓ [B, 1024, H/32, W/32]

输出:
    List of features at different scales
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """简单的 Conv-BN-ReLU 块"""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class SimpleBackbone(nn.Module):
    """
    简化 Backbone，用于 Baseline Testing

    Args:
        input_channels: 输入通道数 (3=RGB, 4=RGB+Density)
        pretrained: 是否加载预训练权重 (当前占位，不支持)
    """

    def __init__(self, input_channels: int = 3, pretrained: bool = False):
        super().__init__()

        self.input_channels = input_channels

        if pretrained:
            print(f"[WARN] Pretrained weights not available for {input_channels}-channel backbone")

        # Stem: 7x7 conv, stride=2
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # Stage 1: stride=1
        self.stage1 = nn.Sequential(
            ConvBlock(64, 128, kernel_size=3, stride=1),
        )

        # Stage 2: stride=2
        self.stage2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBlock(128, 256, kernel_size=3, stride=1),
        )

        # Stage 3: stride=2
        self.stage3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBlock(256, 512, kernel_size=3, stride=1),
        )

        # Stage 4: stride=2
        self.stage4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBlock(512, 1024, kernel_size=3, stride=1),
        )

    def forward(self, x: torch.Tensor) -> list:
        """
        Args:
            x: Input tensor [B, C, H, W]

        Returns:
            List of features at different scales:
            - features[0]: [B, 128, H/2, W/2]  (after stem + stage1)
            - features[1]: [B, 256, H/4, W/4]  (after stage2 with MaxPool)
            - features[2]: [B, 512, H/8, W/8]  (after stage3 with MaxPool)
            - features[3]: [B, 1024, H/16, W/16]  (after stage4 with MaxPool)
        """
        x = self.stem(x)  # [B, 64, H/2, W/2]

        f1 = self.stage1(x)  # [B, 128, H/2, W/2] (stride=1, no downsampling)
        f2 = self.stage2(f1)  # [B, 256, H/4, W/4] (MaxPool stride=2)
        f3 = self.stage3(f2)  # [B, 512, H/8, W/8] (MaxPool stride=2)
        f4 = self.stage4(f3)  # [B, 1024, H/16, W/16] (MaxPool stride=2)

        return [f1, f2, f3, f4]

    def count_parameters(self) -> int:
        """计算参数量"""
        return sum(p.numel() for p in self.parameters())

    def extra_repr(self) -> str:
        return f"input_channels={self.input_channels}"


def init_4channel_from_3channel(backbone_4ch: SimpleBackbone, backbone_3ch: SimpleBackbone):
    """
    从 3 通道 backbone 初始化 4 通道 backbone

    策略:
    - RGB 三个通道：复制原权重
    - Density 通道：使用 RGB 权重的均值

    Args:
        backbone_4ch: 4 通道 backbone (待初始化)
        backbone_3ch: 3 通道 backbone (源权重)
    """
    # 初始化 stem.conv
    stem_3ch = backbone_3ch.stem[0].weight  # [64, 3, 7, 7]
    stem_4ch = backbone_4ch.stem[0].weight  # [64, 4, 7, 7]

    # RGB 通道：直接复制
    stem_4ch[:, :3, :, :] = stem_3ch[:, :, :]

    # Density 通道：使用 RGB 均值
    stem_4ch[:, 3, :, :] = stem_3ch.mean(dim=1, keepdim=True)

    # 初始化 BatchNorm (保持默认)
    # backbone_4ch.stem[1] 保持默认初始化


def get_simple_backbone(input_channels: int = 3, pretrained: bool = False) -> SimpleBackbone:
    """
    获取 SimpleBackbone 实例

    Args:
        input_channels: 输入通道数 (3 or 4)
        pretrained: 是否加载预训练权重

    Returns:
        SimpleBackbone 实例
    """
    backbone = SimpleBackbone(input_channels=input_channels, pretrained=pretrained)

    if pretrained and input_channels == 4:
        # 从 3 通道 backbone 初始化
        backbone_3ch = SimpleBackbone(input_channels=3, pretrained=False)
        init_4channel_from_3channel(backbone, backbone_3ch)

    return backbone


__all__ = ["SimpleBackbone", "get_simple_backbone", "init_4channel_from_3channel"]
