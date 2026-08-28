# -*- coding: utf-8 -*-
"""
Encoder 模块

根据申报书 3.2.1 实现雾密度网络的 Encoder 部分。

结构:
    Input RGB [B, 3, H, W]
        ↓
    Conv Block 1 (3 -> base_channels)
        ↓
    Conv Block 2 (base_channels -> base_channels)
        ↓
    Downsample Conv Block (base_channels -> base_channels*2)
        ↓
    F0 [B, base_channels*2, H/2, W/2]

输入：
    [B, 3, H, W]

输出：
    [B, base_channels*2, H/2, W/2]

工程实现参数:
    - base_channels: 基础通道数（默认 32，可配置）
"""

import torch
import torch.nn as nn

from .basic_blocks import ConvBlock, DownsampleBlock


class Encoder(nn.Module):
    """
    雾密度网络 Encoder

    结构:
        Conv Block 1 (3 -> base_channels)
        Conv Block 2 (base_channels -> base_channels)
        Downsample Conv Block (base_channels -> base_channels*2)

    Args:
        base_channels: 基础通道数（工程实现参数，默认 32）
    """

    def __init__(self, base_channels: int = 32):
        super().__init__()

        self.base_channels = base_channels

        # Conv Block 1: 3 -> base_channels
        self.conv1 = ConvBlock(3, base_channels, kernel_size=3)

        # Conv Block 2: base_channels -> base_channels
        self.conv2 = ConvBlock(base_channels, base_channels, kernel_size=3)

        # Downsample: base_channels -> base_channels*2, H*W -> H/2*W/2
        self.downsample = DownsampleBlock(base_channels, base_channels * 2, kernel_size=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入图像 [B, 3, H, W]

        Returns:
            F0: 编码特征 [B, base_channels*2, H/2, W/2]
        """
        # Conv Block 1
        x = self.conv1(x)  # [B, base_channels, H, W]

        # Conv Block 2
        x = self.conv2(x)  # [B, base_channels, H, W]

        # Downsample
        x = self.downsample(x)  # [B, base_channels*2, H/2, W/2]

        return x

    def extra_repr(self) -> str:
        return f"base_channels={self.base_channels}"

    def get_output_shape(self, input_h: int, input_w: int) -> tuple:
        """
        计算输出形状

        Args:
            input_h: 输入高度
            input_w: 输入宽度

        Returns:
            (C, H, W): 输出通道、高度、宽度
        """
        out_h = input_h // 2
        out_w = input_w // 2
        out_c = self.base_channels * 2
        return (out_c, out_h, out_w)

    def count_parameters(self) -> int:
        """计算参数量"""
        return sum(p.numel() for p in self.parameters())
