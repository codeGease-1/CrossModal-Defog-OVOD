# -*- coding: utf-8 -*-
"""
Guided Filtering (引导滤波)

基于 He et al. "Guided Image Filtering" (ECCV 2010) 的 PyTorch 实现。

引用:
    K. He, J. Sun, and X. Tang, "Guided Image Filtering," ECCV 2010.

输入:
    guide: 引导图 [B, C, H, W] 或 [B, 1, H, W]
    src: 输入图 [B, 1, H, W]
    radius: 滤波半径 r（工程实现参数）
    eps: 正则化参数 ε（工程实现参数）

输出:
    out: 滤波结果 [B, 1, H, W]

注意:
    - 所有计算在 GPU 上进行（如果可用）
    - 设备从输入 tensor 自动继承
    - 不进行 CPU/GPU 拷贝
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _mean_filter(x: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """
    均值滤波（用于计算局部统计量）

    Args:
        x: 输入 tensor [B, C, H, W]
        kernel_size: 滤波核大小 (2*radius + 1)

    Returns:
        均值滤波结果 [B, C, H, W]
    """
    # 使用 padding 保持尺寸不变
    pad = kernel_size // 2
    # 使用 avg_pool2d 进行均值滤波
    return F.avg_pool2d(x, kernel_size=kernel_size, stride=1, padding=pad)


def _get_box_filter(kernel_size: int, device: torch.device) -> torch.Tensor:
    """
    创建 box filter 核

    Args:
        kernel_size: 核大小
        device: 设备

    Returns:
        box kernel [1, 1, kernel_size, kernel_size]
    """
    k = torch.ones(1, 1, kernel_size, kernel_size, device=device)
    return k / (kernel_size * kernel_size)


def guided_filter(
    guide: torch.Tensor,
    src: torch.Tensor,
    radius: int = 15,
    eps: float = 0.01,
) -> torch.Tensor:
    """
    Guided Filtering

    引导滤波用于在保持边缘的同时平滑图像。

    Args:
        guide: 引导图 [B, C, H, W] 或 [B, 1, H, W]
        src: 输入图 [B, 1, H, W]
        radius: 滤波半径 r（工程实现参数，默认 15）
        eps: 正则化参数 ε（工程实现参数，默认 0.01）

    Returns:
        out: 滤波结果 [B, 1, H, W]

    参考公式:
        q = a_k * I + b_k
        其中:
            a_k = (cov_Ip(I, p) / (var_Ik(I) + eps^2))
            b_k = E(p) - a_k * E(I)
    """
    # 检查输入形状
    if src.dim() != 4:
        raise ValueError(f"src must be 4D tensor, got {src.dim()}D")

    if guide.dim() != 4:
        raise ValueError(f"guide must be 4D tensor, got {guide.dim()}D")

    if src.shape[0] != guide.shape[0]:
        raise ValueError(f"batch size mismatch: src={src.shape[0]}, guide={guide.shape[0]}")

    if src.shape[2:] != guide.shape[2:]:
        raise ValueError(f"spatial size mismatch: src={src.shape[2:]}, guide={guide.shape[2:]}")

    # 获取设备
    device = src.device
    dtype = src.dtype

    B, _, H, W = src.shape

    # 如果引导图是多通道，转换为灰度（取均值）
    if guide.shape[1] > 1:
        guide = guide.mean(dim=1, keepdim=True)

    # 确保 src 是单通道
    if src.shape[1] > 1:
        src = src.mean(dim=1, keepdim=True)

    # kernel_size = 2 * radius + 1
    kernel_size = 2 * radius + 1

    # 计算引导图的局部统计量
    # E(I): 引导图均值
    mean_guide = _mean_filter(guide, kernel_size)
    # E(I^2): 引导图平方均值
    mean_guide_sq = _mean_filter(guide * guide, kernel_size)
    # var(I): 引导图方差 = E(I^2) - E(I)^2
    var_guide = mean_guide_sq - mean_guide * mean_guide

    # 计算 src 的局部均值
    # E(p): src 均值
    mean_src = _mean_filter(src, kernel_size)
    # E(Ip): 引导图与 src 的乘积均值
    mean_guide_src = _mean_filter(guide * src, kernel_size)
    # cov(I, p): 协方差 = E(Ip) - E(I)E(p)
    cov_guide_src = mean_guide_src - mean_guide * mean_src

    # 计算 a 和 b
    # a = cov(I, p) / (var(I) + eps)
    a = cov_guide_src / (var_guide + eps)
    # b = E(p) - a * E(I)
    b = mean_src - a * mean_guide

    # 对 a 和 b 进行均值滤波
    mean_a = _mean_filter(a, kernel_size)
    mean_b = _mean_filter(b, kernel_size)

    # 输出 q = a * I + b
    out = mean_a * guide + mean_b

    # 裁剪到 [0, 1] 范围（数值稳定性）
    out = torch.clamp(out, 0.0, 1.0)

    return out


class GuidedFilter(nn.Module):
    """
    Guided Filtering 模块（nn.Module 版本）

    用于在神经网络中作为可配置模块使用。
    """

    def __init__(
        self,
        radius: int = 15,
        eps: float = 0.01,
    ):
        """
        Args:
            radius: 滤波半径 r（工程实现参数，默认 15）
            eps: 正则化参数 ε（工程实现参数，默认 0.01）
        """
        super().__init__()
        self.radius = radius
        self.eps = eps

    def forward(
        self,
        guide: torch.Tensor,
        src: torch.Tensor,
    ) -> torch.Tensor:
        return guided_filter(guide, src, self.radius, self.eps)

    def extra_repr(self) -> str:
        return f"radius={self.radius}, eps={self.eps}"
