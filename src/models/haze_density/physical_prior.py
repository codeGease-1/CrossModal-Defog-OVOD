# -*- coding: utf-8 -*-
"""
物理先验雾密度估计模块

严格依据《最终版申报书.pdf》3.2.1 节实现。

输入:
    image: 含雾 RGB 图像 [B, 3, H, W], 范围 [0, 1]

输出:
    s_final: 雾密度图 [B, 1, H, W], 范围 [0, 1]

处理流程:
    1. Dark Channel Prior (DCP)
    2. Local Contrast Prior (LCP)
    3. Color Shift Prior (CSP)
    4. Normalization (min-max)
    5. Weighted Fusion: S = 0.5*D + 0.3*C + 0.2*K
    6. Power Enhancement: S_hat = S^1.5
    7. Guided Filtering: S_final = GF(G, S_hat)

申报书规定参数（不可修改）:
    - weight_dark: 0.5
    - weight_contrast: 0.3
    - weight_color: 0.2
    - exponent_mu: 1.5

工程实现参数（可调整）:
    - window_size: 局部窗口大小（默认 15）
    - guided_radius: 引导滤波半径（默认 15）
    - guided_eps: 引导滤波正则化（默认 0.01）
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional, Dict

from .guided_filter import guided_filter


# ============================================================================
# 申报书规定参数（不可修改）
# ============================================================================

# 加权融合权重
WEIGHT_DARK = 0.5
WEIGHT_CONTRAST = 0.3
WEIGHT_COLOR = 0.2

# 非线性指数
EXPONENT_MU = 1.5


# ============================================================================
# 工具函数
# ============================================================================


def _local_min(
    x: torch.Tensor,
    window_size: int,
) -> torch.Tensor:
    """
    局部最小值（用于 Dark Channel）

    Args:
        x: 输入 [B, C, H, W]
        window_size: 窗口大小（奇数）

    Returns:
        局部最小值 [B, C, H, W]
    """
    # 使用 max_pool2d 的负值技巧实现 min_pool
    pad = window_size // 2
    x_neg = -x
    min_val = -F.max_pool2d(x_neg, kernel_size=window_size, stride=1, padding=pad)
    return min_val


def _local_mean(
    x: torch.Tensor,
    window_size: int,
) -> torch.Tensor:
    """
    局部均值

    Args:
        x: 输入 [B, C, H, W]
        window_size: 窗口大小（奇数）

    Returns:
        局部均值 [B, C, H, W]
    """
    pad = window_size // 2
    return F.avg_pool2d(x, kernel_size=window_size, stride=1, padding=pad)


def min_max_normalize(
    x: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Min-Max 归一化到 [0, 1]

    Args:
        x: 输入 tensor（任意形状）
        eps: 小常数，防止除零

    Returns:
        归一化结果 [0, 1]
    """
    x_min = x.min(dim=(2, 3), keepdim=True)[0]  # [B, C, 1, 1]
    x_max = x.max(dim=(2, 3), keepdim=True)[0]  # [B, C, 1, 1]

    x_range = x_max - x_min + eps
    x_norm = (x - x_min) / x_range

    # 裁剪到 [0, 1]
    x_norm = torch.clamp(x_norm, 0.0, 1.0)

    return x_norm


# ============================================================================
# 物理先验计算
# ============================================================================


def dark_channel(
    image: torch.Tensor,
    window_size: int = 15,
) -> torch.Tensor:
    """
    Dark Channel Prior (暗通道先验)

    公式:
        D(x) = min_{y∈Ω(x)} min_{c∈{r,g,b}} I_c(y)

    Args:
        image: RGB 图像 [B, 3, H, W], 范围 [0, 1]
        window_size: 局部窗口大小（工程实现参数，默认 15）

    Returns:
        dark_channel: 暗通道图 [B, 1, H, W]

    参考:
        He, K., Sun, J., & Tang, X. (2009). Single image haze removal using dark channel prior.
    """
    if image.shape[1] != 3:
        raise ValueError(f"image must have 3 channels, got {image.shape[1]}")

    # 在空间域上取局部最小值
    # 先对每个通道做局部 min
    local_min_per_channel = _local_min(image, window_size)  # [B, 3, H, W]

    # 再在通道维度上取 min
    dark_channel_map, _ = torch.min(local_min_per_channel, dim=1, keepdim=True)  # [B, 1, H, W]

    return dark_channel_map


def local_contrast(
    image: torch.Tensor,
    window_size: int = 15,
) -> torch.Tensor:
    """
    Local Contrast Prior (局部对比度先验)

    公式:
        G = 0.299 * Ir + 0.587 * Ig + 0.114 * Ib
        C(x) = |G(x) - G_avg(x)|
        C_hat = 1 - N(C)

    Args:
        image: RGB 图像 [B, 3, H, W], 范围 [0, 1]
        window_size: 局部窗口大小（工程实现参数，默认 15）

    Returns:
        local_contrast: 局部对比度图 [B, 1, H, W]
    """
    if image.shape[1] != 3:
        raise ValueError(f"image must have 3 channels, got {image.shape[1]}")

    # 分离 RGB 通道
    ir = image[:, 0:1, :, :]  # [B, 1, H, W]
    ig = image[:, 1:2, :, :]
    ib = image[:, 2:3, :, :]

    # 转换为灰度图（ITU-R BT.601 标准）
    gray = 0.299 * ir + 0.587 * ig + 0.114 * ib  # [B, 1, H, W]

    # 计算局部均值
    gray_avg = _local_mean(gray, window_size)  # [B, 1, H, W]

    # 计算对比度：绝对差值
    contrast = torch.abs(gray - gray_avg)  # [B, 1, H, W]

    # 归一化
    contrast_norm = min_max_normalize(contrast)  # [B, 1, H, W]

    # 逆归一化：1 - N(C)
    local_contrast_map = 1.0 - contrast_norm

    # 裁剪到 [0, 1]
    local_contrast_map = torch.clamp(local_contrast_map, 0.0, 1.0)

    return local_contrast_map


def color_shift(
    image: torch.Tensor,
    window_size: int = 15,
) -> torch.Tensor:
    """
    Color Shift Prior (颜色偏移先验)

    公式:
        K = |Ir - Ig| + |Ir - Ib|
        K_hat = 1 - N(K)

    Args:
        image: RGB 图像 [B, 3, H, W], 范围 [0, 1]
        window_size: 局部窗口大小（工程实现参数，默认 15）

    Returns:
        color_shift: 颜色偏移图 [B, 1, H, W]

    注意:
        窗口大小在此函数中保留作为接口一致性，但实际计算不使用。
        颜色偏移是逐像素计算，不需要局部窗口。
    """
    if image.shape[1] != 3:
        raise ValueError(f"image must have 3 channels, got {image.shape[1]}")

    # 分离 RGB 通道
    ir = image[:, 0:1, :, :]  # [B, 1, H, W]
    ig = image[:, 1:2, :, :]
    ib = image[:, 2:3, :, :]

    # 计算颜色偏移
    color_shift_map = torch.abs(ir - ig) + torch.abs(ir - ib)  # [B, 1, H, W]

    # 归一化
    color_shift_norm = min_max_normalize(color_shift_map)  # [B, 1, H, W]

    # 逆归一化：1 - N(K)
    color_shift_map = 1.0 - color_shift_norm

    # 裁剪到 [0, 1]
    color_shift_map = torch.clamp(color_shift_map, 0.0, 1.0)

    return color_shift_map


# ============================================================================
# 融合与后处理
# ============================================================================


def compute_physical_prior(
    image: torch.Tensor,
    window_size: int = 15,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    计算并融合三种物理先验

    公式:
        S_hat = (w_D * D_hat + w_C * C_hat + w_K * K_hat)^μ

    其中:
        w_D = 0.5 (申报书规定)
        w_C = 0.3 (申报书规定)
        w_K = 0.2 (申报书规定)
        μ = 1.5 (申报书规定)

    Args:
        image: RGB 图像 [B, 3, H, W], 范围 [0, 1]
        window_size: 局部窗口大小（工程实现参数，默认 15）

    Returns:
        D_hat: 归一化暗通道 [B, 1, H, W]
        C_hat: 归一化局部对比度 [B, 1, H, W]
        K_hat: 归一化颜色偏移 [B, 1, H, W]
        S_hat: 融合后的雾密度图 [B, 1, H, W]
    """
    # 计算三种先验
    d_raw = dark_channel(image, window_size)  # [B, 1, H, W]
    c_raw = local_contrast(image, window_size)  # [B, 1, H, W]
    k_raw = color_shift(image, window_size)  # [B, 1, H, W]

    # 归一化（local_contrast 和 color_shift 内部已归一化，但 dark_channel 需要）
    d_hat = min_max_normalize(d_raw)  # [B, 1, H, W]
    c_hat = c_raw  # 已经是归一化后的 [0, 1]
    k_hat = k_raw  # 已经是归一化后的 [0, 1]

    # 加权融合
    # S = 0.5 * D_hat + 0.3 * C_hat + 0.2 * K_hat
    s_weighted = (
        WEIGHT_DARK * d_hat
        + WEIGHT_CONTRAST * c_hat
        + WEIGHT_COLOR * k_hat
    )  # [B, 1, H, W]

    # 裁剪到 [0, 1]（理论上加权后应在 [0, 1] 内，但为防止数值误差）
    s_weighted = torch.clamp(s_weighted, 0.0, 1.0)

    # 非线性增强：S_hat = S^μ
    s_hat = torch.pow(s_weighted, EXPONENT_MU)  # [B, 1, H, W]

    return d_hat, c_hat, k_hat, s_hat


def generate_s_final(
    image: torch.Tensor,
    window_size: int = 15,
    guided_radius: int = 15,
    guided_eps: float = 0.01,
    return_intermediate: bool = False,
) -> torch.Tensor:
    """
    生成最终雾密度图 S_final

    完整流程:
        1. 计算三种物理先验
        2. 归一化
        3. 加权融合
        4. 非线性增强
        5. Guided Filtering

    Args:
        image: RGB 图像 [B, 3, H, W], 范围 [0, 1]
        window_size: 局部窗口大小（工程实现参数，默认 15）
        guided_radius: 引导滤波半径（工程实现参数，默认 15）
        guided_eps: 引导滤波正则化（工程实现参数，默认 0.01）
        return_intermediate: 是否返回中间结果（用于 debug/可视化）

    Returns:
        如果 return_intermediate=False:
            s_final: 最终雾密度图 [B, 1, H, W]
        如果 return_intermediate=True:
            dict 包含:
                - D_hat: 归一化暗通道
                - C_hat: 归一化局部对比度
                - K_hat: 归一化颜色偏移
                - S_hat: 融合后（Guided Filter 前）
                - S_final: 最终雾密度图
    """
    # 输入检查
    if image.dim() != 4:
        raise ValueError(f"image must be 4D tensor [B, 3, H, W], got {image.dim()}D")

    if image.shape[1] != 3:
        raise ValueError(f"image must have 3 channels, got {image.shape[1]}")

    # 检查输入范围
    if image.min() < 0 or image.max() > 1:
        # 警告但不强制，因为用户可能已经处理过
        pass

    # 计算物理先验
    d_hat, c_hat, k_hat, s_hat = compute_physical_prior(image, window_size)

    # 计算灰度图作为引导图
    ir = image[:, 0:1, :, :]
    ig = image[:, 1:2, :, :]
    ib = image[:, 2:3, :, :]
    gray_guide = 0.299 * ir + 0.587 * ig + 0.114 * ib  # [B, 1, H, W]

    # Guided Filtering
    s_final = guided_filter(
        guide=gray_guide,
        src=s_hat,
        radius=guided_radius,
        eps=guided_eps,
    )

    if return_intermediate:
        return {
            "D_hat": d_hat,
            "C_hat": c_hat,
            "K_hat": k_hat,
            "S_hat": s_hat,
            "S_final": s_final,
        }

    return s_final


class PhysicalPriorModule(torch.nn.Module):
    """
    物理先验模块（nn.Module 版本）

    用于在神经网络中作为可配置模块使用。
    """

    def __init__(
        self,
        window_size: int = 15,
        guided_radius: int = 15,
        guided_eps: float = 0.01,
    ):
        """
        Args:
            window_size: 局部窗口大小（工程实现参数，默认 15）
            guided_radius: 引导滤波半径（工程实现参数，默认 15）
            guided_eps: 引导滤波正则化（工程实现参数，默认 0.01）
        """
        super().__init__()
        self.window_size = window_size
        self.guided_radius = guided_radius
        self.guided_eps = guided_eps

    def forward(
        self,
        image: torch.Tensor,
        return_intermediate: bool = False,
    ):
        return generate_s_final(
            image,
            window_size=self.window_size,
            guided_radius=self.guided_radius,
            guided_eps=self.guided_eps,
            return_intermediate=return_intermediate,
        )

    def extra_repr(self) -> str:
        return (
            f"window_size={self.window_size}, "
            f"guided_radius={self.guided_radius}, "
            f"guided_eps={self.guided_eps}"
        )
