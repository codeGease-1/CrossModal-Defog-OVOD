# -*- coding: utf-8 -*-
"""
基础卷积块模块

提供雾密度网络使用的基础构建块：
- ConvBlock: Conv + InstanceNorm + ReLU
- DownsampleBlock: 下采样块
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """
    基础卷积块

    结构：Conv -> InstanceNorm -> ReLU

    Args:
        in_channels: 输入通道数
        out_channels: 输出通道数
        kernel_size: 卷积核大小（默认 3）
        stride: 步长（默认 1）
        padding: 填充（默认根据 kernel_size 自动计算）
        use_norm: 是否使用 InstanceNorm（默认 True）
        use_relu: 是否使用 ReLU（默认 True）
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = None,
        use_norm: bool = True,
        use_relu: bool = True,
    ):
        super().__init__()

        if padding is None:
            padding = kernel_size // 2

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=not use_norm,
        )

        self.norm = nn.InstanceNorm2d(out_channels) if use_norm else nn.Identity()
        self.relu = nn.ReLU(inplace=True) if use_relu else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        return x

    def extra_repr(self) -> str:
        return (
            f"in_channels={self.conv.in_channels}, "
            f"out_channels={self.conv.out_channels}, "
            f"kernel_size={self.conv.kernel_size[0]}, "
            f"stride={self.conv.stride[0]}"
        )


class DownsampleBlock(nn.Module):
    """
    下采样块

    结构：Conv(stride=2) -> InstanceNorm -> ReLU

    Args:
        in_channels: 输入通道数
        out_channels: 输出通道数
        kernel_size: 卷积核大小（默认 3）
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
    ):
        super().__init__()

        padding = kernel_size // 2

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=2,
            padding=padding,
            bias=False,
        )
        self.norm = nn.InstanceNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        return x

    def extra_repr(self) -> str:
        return (
            f"in_channels={self.conv.in_channels}, "
            f"out_channels={self.conv.out_channels}"
        )


class DoubleConvBlock(nn.Module):
    """
    双卷积块

    结构：ConvBlock -> ConvBlock

    Args:
        in_channels: 输入通道数
        out_channels: 输出通道数
        kernel_size: 卷积核大小（默认 3）
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
    ):
        super().__init__()

        self.conv1 = ConvBlock(in_channels, out_channels, kernel_size)
        self.conv2 = ConvBlock(out_channels, out_channels, kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        return x

    def extra_repr(self) -> str:
        return (
            f"in_channels={self.conv1.conv.in_channels}, "
            f"out_channels={self.conv2.conv.out_channels}"
        )
