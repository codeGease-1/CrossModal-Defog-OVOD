# -*- coding: utf-8 -*-
"""
残差块模块

提供：
- ResidualBlock (RB): 标准残差块
- DilatedResidualBlock (SDRB): 膨胀残差块（申报书规定）
"""

import torch
import torch.nn as nn

from .basic_blocks import ConvBlock


class ResidualBlock(nn.Module):
    """
    标准残差块 (RB)

    结构：
        x
        ↓
        ConvBlock -> ConvBlock
        ↓              ↓
        +--------------+
        ↓
        Output

    输入输出 shape 一致。

    Args:
        channels: 通道数
        kernel_size: 卷积核大小（默认 3）
    """

    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()

        self.conv1 = ConvBlock(channels, channels, kernel_size)
        self.conv2 = ConvBlock(channels, channels, kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入 [B, C, H, W]

        Returns:
            输出 [B, C, H, W]
        """
        residual = x

        out = self.conv1(x)
        out = self.conv2(out)

        out = out + residual
        return out

    def extra_repr(self) -> str:
        return f"channels={self.conv1.conv.in_channels}"


class DilatedResidualBlock(nn.Module):
    """
    膨胀残差块 (SDRB) - Residual Smooth Dilated Convolution Block

    根据申报书 3.2.1 实现。

    结构：
        x
        ↓
        Conv(dilated) -> InstanceNorm -> ReLU
        ↓
        Conv(dilated) -> InstanceNorm -> ReLU
        ↓              ↓
        +--------------+
        ↓
        Output

    特点：
        - 使用膨胀卷积扩大感受野
        - 保持输入输出尺寸一致
        - residual connection

    Args:
        channels: 通道数
        dilation: 膨胀率（申报书规定：2, 3, 4）
        kernel_size: 卷积核大小（默认 3）
    """

    def __init__(
        self,
        channels: int,
        dilation: int = 2,
        kernel_size: int = 3,
    ):
        super().__init__()

        # 计算 padding 以保持输出尺寸不变
        # padding = (kernel_size - 1) * dilation / 2
        padding = (kernel_size - 1) * dilation // 2

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=kernel_size,
                dilation=dilation,
                padding=padding,
                bias=False,
            ),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=kernel_size,
                dilation=dilation,
                padding=padding,
                bias=False,
            ),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
        )

        self.dilation = dilation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入 [B, C, H, W]

        Returns:
            输出 [B, C, H, W]
        """
        residual = x

        out = self.conv1(x)
        out = self.conv2(out)

        out = out + residual
        return out

    def extra_repr(self) -> str:
        return f"channels={self.conv1[0].in_channels}, dilation={self.dilation}"


class SDRB(DilatedResidualBlock):
    """
    SDRB 别名，方便使用。

    与 DilatedResidualBlock 完全相同。
    """

    pass
