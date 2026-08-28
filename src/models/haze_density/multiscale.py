# -*- coding: utf-8 -*-
"""
多尺度特征提取模块

根据申报书 3.2.1 实现雾密度网络的多尺度分支。

结构:
    Input F0 [B, C, H, W]
        ↓
    Branch 1 (dilation=2):
        SDRB(r=2) → RB → RB → ECA
        ↓
    Branch 2 (dilation=3):
        SDRB(r=3) → RB → RB → ECA
        ↓
    Branch 3 (dilation=4):
        SDRB(r=4) → RB → RB → ECA
        ↓
    Concat: [B, 3*C, H, W]

输入：
    F0: [B, C, H, W]

输出：
    F1, F2, F3: 三个分支的输出 [B, C, H, W]
    F_concat: 拼接后的特征 [B, 3*C, H, W]

申报书规定参数:
    - dilation_rates: [2, 3, 4]
"""

import torch
import torch.nn as nn

from .residual_blocks import ResidualBlock, DilatedResidualBlock
from .eca import ECA


class MultiScaleBranch(nn.Module):
    """
    单一路多尺度分支

    结构：SDRB → RB → RB → ECA

    Args:
        channels: 通道数
        dilation: 膨胀率（2, 3, 或 4）
    """

    def __init__(self, channels: int, dilation: int):
        super().__init__()

        self.dilation = dilation

        # SDRB (Dilated Residual Block)
        self.sdrb = DilatedResidualBlock(channels, dilation=dilation)

        # RB (标准残差块) x2
        self.rb1 = ResidualBlock(channels)
        self.rb2 = ResidualBlock(channels)

        # ECA (Efficient Channel Attention)
        self.eca = ECA(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入 [B, C, H, W]

        Returns:
            输出 [B, C, H, W]
        """
        # SDRB
        out = self.sdrb(x)

        # RB x2
        out = self.rb1(out)
        out = self.rb2(out)

        # ECA
        out = self.eca(out)

        return out

    def extra_repr(self) -> str:
        return f"channels={self.sdrb.conv1[0].in_channels}, dilation={self.dilation}"


class MultiScaleFeatureExtractor(nn.Module):
    """
    多尺度特征提取器

    根据申报书 3.2.1 实现三个并行分支。

    结构:
        Branch 1 (dilation=2): SDRB → RB → RB → ECA
        Branch 2 (dilation=3): SDRB → RB → RB → ECA
        Branch 3 (dilation=4): SDRB → RB → RB → ECA

    Args:
        channels: 输入通道数
        dilation_rates: 膨胀率列表（申报书规定：[2, 3, 4]）
    """

    def __init__(
        self,
        channels: int,
        dilation_rates: list = None,
    ):
        super().__init__()

        if dilation_rates is None:
            # 申报书规定参数
            dilation_rates = [2, 3, 4]

        self.channels = channels
        self.dilation_rates = dilation_rates

        # 三个并行分支
        self.branch1 = MultiScaleBranch(channels, dilation_rates[0])
        self.branch2 = MultiScaleBranch(channels, dilation_rates[1])
        self.branch3 = MultiScaleBranch(channels, dilation_rates[2])

    def forward(
        self,
        x: torch.Tensor,
        return_separate: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            x: 输入 [B, C, H, W]
            return_separate: 是否分别返回三个分支的输出

        Returns:
            如果 return_separate=False:
                F_concat: [B, 3*C, H, W]
            如果 return_separate=True:
                (F1, F2, F3, F_concat)
        """
        # 三个分支并行处理
        f1 = self.branch1(x)  # [B, C, H, W]
        f2 = self.branch2(f1)  # [B, C, H, W] - 注意：串联处理
        f3 = self.branch3(f2)  # [B, C, H, W]

        # 拼接
        f_concat = torch.cat([f1, f2, f3], dim=1)  # [B, 3*C, H, W]

        if return_separate:
            return f1, f2, f3, f_concat

        return f_concat

    def extra_repr(self) -> str:
        return f"channels={self.channels}, dilation_rates={self.dilation_rates}"


class ParallelMultiScaleFeatureExtractor(nn.Module):
    """
    并行多尺度特征提取器（三个独立分支）

    三个分支独立处理输入，然后拼接。

    结构:
        Input F0
            ↓
        Branch 1 (dilation=2): SDRB → RB → RB → ECA
        Branch 2 (dilation=3): SDRB → RB → RB → ECA
        Branch 3 (dilation=4): SDRB → RB → RB → ECA
            ↓
        Concat

    Args:
        channels: 输入通道数
        dilation_rates: 膨胀率列表（申报书规定：[2, 3, 4]）
    """

    def __init__(
        self,
        channels: int,
        dilation_rates: list = None,
    ):
        super().__init__()

        if dilation_rates is None:
            # 申报书规定参数
            dilation_rates = [2, 3, 4]

        self.channels = channels
        self.dilation_rates = dilation_rates

        # 三个独立分支
        self.branch1 = MultiScaleBranch(channels, dilation_rates[0])
        self.branch2 = MultiScaleBranch(channels, dilation_rates[1])
        self.branch3 = MultiScaleBranch(channels, dilation_rates[2])

    def forward(
        self,
        x: torch.Tensor,
        return_separate: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            x: 输入 [B, C, H, W]
            return_separate: 是否分别返回三个分支的输出

        Returns:
            如果 return_separate=False:
                F_concat: [B, 3*C, H, W]
            如果 return_separate=True:
                (F1, F2, F3, F_concat)
        """
        # 三个分支并行处理（独立输入）
        f1 = self.branch1(x)  # [B, C, H, W]
        f2 = self.branch2(x)  # [B, C, H, W]
        f3 = self.branch3(x)  # [B, C, H, W]

        # 拼接
        f_concat = torch.cat([f1, f2, f3], dim=1)  # [B, 3*C, H, W]

        if return_separate:
            return f1, f2, f3, f_concat

        return f_concat

    def extra_repr(self) -> str:
        return f"channels={self.channels}, dilation_rates={self.dilation_rates}"

    def count_parameters(self) -> int:
        """计算参数量"""
        return sum(p.numel() for p in self.parameters())
