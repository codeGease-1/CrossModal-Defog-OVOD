# -*- coding: utf-8 -*-
"""
特征融合模块

根据申报书 3.2.1 实现雾密度网络的 Fusion 部分。

结构:
    Input F1, F2, F3 [B, C, H, W] each
        ↓
    Concat (channel dimension)
        ↓
    F_concat [B, 3*C, H, W]
        ↓
    3×3 Conv (3*C -> C)
        ↓
    ECA
        ↓
    F_fuse [B, C, H, W]

输入：
    F1, F2, F3: 三个分支的输出 [B, C, H, W]

输出：
    F_fuse: 融合后的特征 [B, C, H, W]
"""

import torch
import torch.nn as nn

from .basic_blocks import ConvBlock
from .eca import ECA


class FusionModule(nn.Module):
    """
    特征融合模块

    结构:
        Concat (channel dimension)
        3×3 Conv
        ECA

    Args:
        in_channels: 每个分支的通道数
        out_channels: 输出通道数（默认与 in_channels 相同）
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int = None,
    ):
        super().__init__()

        if out_channels is None:
            out_channels = in_channels

        self.in_channels = in_channels
        self.out_channels = out_channels

        # Concat 后通道数为 3 * in_channels
        concat_channels = in_channels * 3

        # 3×3 Conv
        self.conv = ConvBlock(
            concat_channels,
            out_channels,
            kernel_size=3,
        )

        # ECA
        self.eca = ECA(out_channels)

    def forward(
        self,
        f1: torch.Tensor,
        f2: torch.Tensor,
        f3: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            f1: 分支 1 输出 [B, C, H, W]
            f2: 分支 2 输出 [B, C, H, W]
            f3: 分支 3 输出 [B, C, H, W]

        Returns:
            F_fuse: 融合后的特征 [B, out_channels, H, W]
        """
        # 检查输入形状
        assert f1.shape == f2.shape == f3.shape, "Input features must have same shape"
        assert f1.shape[1] == self.in_channels, f"Expected {self.in_channels} channels, got {f1.shape[1]}"

        # Concat (channel dimension)
        f_concat = torch.cat([f1, f2, f3], dim=1)  # [B, 3*C, H, W]

        # 3×3 Conv
        f_out = self.conv(f_concat)  # [B, out_channels, H, W]

        # ECA
        f_fuse = self.eca(f_out)  # [B, out_channels, H, W]

        return f_fuse

    def extra_repr(self) -> str:
        return f"in_channels={self.in_channels}, out_channels={self.out_channels}"

    def count_parameters(self) -> int:
        """计算参数量"""
        return sum(p.numel() for p in self.parameters())
