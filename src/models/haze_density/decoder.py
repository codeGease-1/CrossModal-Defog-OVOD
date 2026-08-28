# -*- coding: utf-8 -*-
"""
Decoder 模块

根据申报书 3.2.1 实现雾密度网络的 Decoder 部分。

结构:
    F_fuse [B, C, H/2, W/2]
        ↓
    DeConv / Transposed Convolution (upsampling)
        ↓
    Conv
        ↓
    Conv
        ↓
    Sigmoid (工程实现决策，保证输出在 [0,1])
        ↓
    Haze Density Map [B, 1, H, W]

输入：
    F_fuse: 融合特征 [B, C, H/2, W/2]

输出：
    I_h: 雾密度图 [B, 1, H, W]，范围 [0, 1]

工程实现参数:
    - in_channels: 输入通道数（默认 64，与 Encoder 输出一致）
    - use_sigmoid: 是否使用 Sigmoid 激活（默认 True，工程实现决策）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Decoder(nn.Module):
    """
    雾密度网络 Decoder

    结构:
        DeConv (upsampling 2x)
        Conv
        Conv
        Sigmoid (可选)

    Args:
        in_channels: 输入通道数（工程实现参数，默认 64）
        use_sigmoid: 是否使用 Sigmoid 激活（工程实现决策，默认 True）
    """

    def __init__(
        self,
        in_channels: int = 64,
        use_sigmoid: bool = True,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.use_sigmoid = use_sigmoid

        # DeConv / Transposed Convolution (upsampling 2x)
        # 从 [B, C, H/2, W/2] 到 [B, C/2, H, W]
        self.deconv = nn.ConvTranspose2d(
            in_channels,
            in_channels // 2,
            kernel_size=2,
            stride=2,
            padding=0,
            bias=False,
        )
        self.norm1 = nn.InstanceNorm2d(in_channels // 2)
        self.relu1 = nn.ReLU(inplace=True)

        # Conv 1
        self.conv1 = nn.Conv2d(
            in_channels // 2,
            in_channels // 4,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.norm2 = nn.InstanceNorm2d(in_channels // 4)
        self.relu2 = nn.ReLU(inplace=True)

        # Conv 2 (输出 1 通道)
        self.conv2 = nn.Conv2d(
            in_channels // 4,
            1,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.norm3 = nn.InstanceNorm2d(1)
        self.relu3 = nn.ReLU(inplace=True)

        # Sigmoid activation (工程实现决策，保证输出在 [0,1])
        self.sigmoid = nn.Sigmoid() if use_sigmoid else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入特征 [B, C, H/2, W/2]

        Returns:
            I_h: 雾密度图 [B, 1, H, W]，范围 [0, 1]
        """
        # DeConv (upsampling 2x)
        x = self.deconv(x)  # [B, C/2, H, W]
        x = self.norm1(x)
        x = self.relu1(x)

        # Conv 1
        x = self.conv1(x)  # [B, C/4, H, W]
        x = self.norm2(x)
        x = self.relu2(x)

        # Conv 2
        x = self.conv2(x)  # [B, 1, H, W]
        x = self.norm3(x)
        x = self.relu3(x)

        # Sigmoid (工程实现决策)
        if self.use_sigmoid:
            x = self.sigmoid(x)  # [B, 1, H, W], range [0, 1]

        return x

    def extra_repr(self) -> str:
        return f"in_channels={self.in_channels}, use_sigmoid={self.use_sigmoid}"

    def count_parameters(self) -> int:
        """计算参数量"""
        return sum(p.numel() for p in self.parameters())


class DecoderV2(nn.Module):
    """
    Decoder 简化版本

    使用更简洁的结构，参数量更少。

    结构:
        DeConv (upsampling 2x)
        Conv
        Conv (output 1 channel)
        Sigmoid
    """

    def __init__(
        self,
        in_channels: int = 64,
        use_sigmoid: bool = True,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.use_sigmoid = use_sigmoid

        # DeConv (upsampling 2x)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                in_channels // 2,
                kernel_size=2,
                stride=2,
                bias=False,
            ),
            nn.InstanceNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
        )

        # Conv 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels // 2,
                in_channels // 4,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.InstanceNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
        )

        # Conv 2 (output 1 channel)
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels // 4,
                1,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.InstanceNorm2d(1),
            nn.ReLU(inplace=True),
        )

        # Sigmoid
        self.sigmoid = nn.Sigmoid() if use_sigmoid else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.deconv(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sigmoid(x)
        return x

    def extra_repr(self) -> str:
        return f"in_channels={self.in_channels}, use_sigmoid={self.use_sigmoid}"

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
