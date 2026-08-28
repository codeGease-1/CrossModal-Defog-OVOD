# -*- coding: utf-8 -*-
"""
ECA (Efficient Channel Attention) 模块

根据 Wu et al. "EMBD: Efficient Channel Attention for Deep Neural Networks" 实现。

输入：
    [B, C, H, W]

输出：
    [B, C, H, W]

特点:
    - 不改变空间尺寸
    - 不改变 channel 数
    - 自适应 kernel size
    - GPU 兼容

参考:
    Wu, T., et al. "EMC: Efficient Multi-scale Convolution for Channel Attention."
    或
    Wang, Q., et al. "ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks."
"""

import torch
import torch.nn as nn
import math


class ECA(nn.Module):
    """
    Efficient Channel Attention (ECA)

    结构:
        1. Global Average Pooling: [B, C, H, W] -> [B, C, 1, 1]
        2. 1D Conv (adaptive kernel size): [B, C, 1] -> [B, C, 1]
        3. Sigmoid: [B, C, 1] -> [B, C, 1]
        4. Channel-wise scaling: [B, C, H, W] * [B, C, 1, 1]

    Args:
        channels: 输入通道数
        kernel_size: 1D 卷积核大小（默认 None，自动计算）
        scale: gamma 参数，用于计算自适应 kernel size（默认 2.5）
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = None,
        scale: float = 2.5,
    ):
        super().__init__()

        # 计算自适应 kernel size
        if kernel_size is None:
            # gamma = 2.5, b = 1 (odd kernel size)
            # kernel_size = (log_2(C) + b) / gamma | 2
            # 简化为：kernel_size = (log2(C) + 1) / 2.5，取奇数
            kernel_size = max(1, int((math.log2(channels) + 1) / scale))
            # 确保是奇数
            if kernel_size % 2 == 0:
                kernel_size += 1

        self.channels = channels
        self.kernel_size = kernel_size

        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d(1)

        # 1D Conv for channel attention
        self.conv = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            bias=False,
        )

        # Sigmoid activation
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入 [B, C, H, W]

        Returns:
            输出 [B, C, H, W]
        """
        b, c, h, w = x.size()

        # Global Average Pooling: [B, C, H, W] -> [B, C, 1, 1]
        y = self.gap(x)  # [B, C, 1, 1]

        # Reshape for 1D Conv: [B, C, 1, 1] -> [B, C, 1]
        y = y.view(b, c, -1)  # [B, C, 1]

        # 1D Conv: [B, C, 1] -> [B, C, 1]
        y = self.conv(y)  # [B, C, 1]

        # Sigmoid: [B, C, 1] -> [B, C, 1]
        y = self.sigmoid(y)  # [B, C, 1]

        # Reshape back: [B, C, 1] -> [B, C, 1, 1]
        y = y.view(b, c, 1, 1)

        # Channel-wise scaling: [B, C, H, W] * [B, C, 1, 1]
        out = x * y.expand_as(x)

        return out

    def extra_repr(self) -> str:
        return f"channels={self.channels}, kernel_size={self.kernel_size}"


class ECAv2(nn.Module):
    """
    ECA 简化版本（更高效）

    与 ECA 功能相同，但代码更简洁。
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = None,
        scale: float = 2.5,
    ):
        super().__init__()

        if kernel_size is None:
            kernel_size = max(1, int((math.log2(channels) + 1) / scale))
            if kernel_size % 2 == 0:
                kernel_size += 1

        self.channels = channels
        self.kernel_size = kernel_size

        self.conv = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            bias=False,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()

        # GAP + reshape
        y = x.mean(dim=(2, 3))  # [B, C]
        y = y.view(b, c, 1)  # [B, C, 1]

        # 1D Conv + Sigmoid
        y = self.sigmoid(self.conv(y))  # [B, C, 1]

        # Channel-wise scaling
        return x * y.view(b, c, 1, 1)

    def extra_repr(self) -> str:
        return f"channels={self.channels}, kernel_size={self.kernel_size}"
