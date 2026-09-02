# -*- coding: utf-8 -*-
"""
Density Concatenation Baseline Model (Stage 6-2)

将 HazeDensityNet 生成的密度图作为额外输入通道，与 RGB 图像拼接后输入 Backbone。

架构:
    Hazy RGB [B, 3, H, W]
        ↓
    Frozen HazeDensityNet (load best.pth)
        ↓
    Density Map [B, 1, H, W]
        ↓
    Concat → [B, 4, H, W]
        ↓
    Backbone (4-channel input)
        ↓
    Features [List of multi-scale features]

设计要点:
1. HazeDensityNet 默认冻结 (requires_grad=False)
2. 密度图在 no_grad 上下文中生成
3. Backbone 第一层支持 4 通道输入
4. 支持从 3 通道 backbone 初始化 4 通道权重
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Dict, Any, List

from ..haze_density import HazeDensityNet
from ..backbone import SimpleBackbone, init_4channel_from_3channel


class DensityConcatModel(nn.Module):
    """
    Density Concatenation Baseline Model

    Args:
        density_checkpoint: HazeDensityNet checkpoint 路径
        freeze_density: 是否冻结 HazeDensityNet (默认 True)
        backbone_input_channels: Backbone 输入通道数 (默认 4)
        density_base_channels: HazeDensityNet base_channels (默认 32)
    """

    def __init__(
        self,
        density_checkpoint: str = "experiments/haze_density/checkpoints/formal/best.pth",
        freeze_density: bool = True,
        backbone_input_channels: int = 4,
        density_base_channels: int = 32,
    ):
        super().__init__()

        self.density_checkpoint = density_checkpoint
        self.freeze_density = freeze_density
        self.backbone_input_channels = backbone_input_channels

        # ========== HazeDensityNet (Frozen) ==========
        print(f"[DensityConcatModel] Loading HazeDensityNet from {density_checkpoint}")
        self.density_net = HazeDensityNet(base_channels=density_base_channels, use_sigmoid=True)

        # 加载 checkpoint
        checkpoint_path = Path(density_checkpoint)
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            self.density_net.load_state_dict(checkpoint['model_state_dict'])
            print(f"[DensityConcatModel] Loaded density checkpoint successfully")
        else:
            raise FileNotFoundError(f"Density checkpoint not found: {density_checkpoint}")

        # 冻结密度网络
        if freeze_density:
            self.density_net.eval()
            for param in self.density_net.parameters():
                param.requires_grad = False
            print(f"[DensityConcatModel] HazeDensityNet frozen (requires_grad=False)")
        else:
            print(f"[DensityConcatModel] HazeDensityNet trainable (requires_grad=True)")

        # ========== Backbone (4-channel input) ==========
        print(f"[DensityConcatModel] Creating {backbone_input_channels}-channel backbone")
        self.backbone = SimpleBackbone(input_channels=backbone_input_channels)

        # 统计参数量
        self._print_parameter_stats()

    def _print_parameter_stats(self):
        """打印参数量统计"""
        total_params = sum(p.numel() for p in self.parameters())
        density_params = sum(p.numel() for p in self.density_net.parameters())
        backbone_params = sum(p.numel() for p in self.backbone.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        print(f"\n[DensityConcatModel] Parameter Statistics:")
        print(f"  Total parameters:      {total_params:,}")
        print(f"  HazeDensityNet:        {density_params:,} (frozen={self.freeze_density})")
        print(f"  Backbone:              {backbone_params:,}")
        print(f"  Trainable parameters:  {trainable_params:,}")
        print()

    def forward(
        self,
        hazy_image: torch.Tensor,
        return_density: bool = False,
    ) -> Dict[str, Any]:
        """
        前向传播

        Args:
            hazy_image: 含雾 RGB 图像 [B, 3, H, W], 范围 [0, 1]
            return_density: 是否返回密度图 (用于调试/可视化)

        Returns:
            dict 包含:
                - features: List of multi-scale features
                - density_map: 密度图 [B, 1, H, W] (如果 return_density=True)
                - concat_input: 拼接后的输入 [B, 4, H, W] (如果 return_density=True)
        """
        # ========== 生成密度图 (Frozen) ==========
        if self.freeze_density:
            with torch.no_grad():
                density_map = self.density_net(hazy_image)
        else:
            density_map = self.density_net(hazy_image)

        # ========== Concat: RGB + Density ==========
        # hazy_image: [B, 3, H, W]
        # density_map: [B, 1, H, W]
        # concat_input: [B, 4, H, W]
        concat_input = torch.cat([hazy_image, density_map], dim=1)

        # ========== Backbone ==========
        features = self.backbone(concat_input)

        result = {'features': features}

        if return_density:
            result['density_map'] = density_map
            result['concat_input'] = concat_input

        return result

    def load_density_checkpoint(self, checkpoint_path: str):
        """
        重新加载密度网络 checkpoint

        Args:
            checkpoint_path: checkpoint 路径
        """
        print(f"[DensityConcatModel] Reloading density checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        self.density_net.load_state_dict(checkpoint['model_state_dict'])

        if self.freeze_density:
            self.density_net.eval()
            for param in self.density_net.parameters():
                param.requires_grad = False

        print(f"[DensityConcatModel] Density checkpoint reloaded")

    def set_freeze_density(self, freeze: bool):
        """
        设置密度网络冻结状态

        Args:
            freeze: True=冻结，False=可训练
        """
        self.freeze_density = freeze

        if freeze:
            self.density_net.eval()
            for param in self.density_net.parameters():
                param.requires_grad = False
        else:
            self.density_net.train()
            for param in self.density_net.parameters():
                param.requires_grad = True

        print(f"[DensityConcatModel] Density network frozen={freeze}")

    def count_parameters(self) -> Dict[str, int]:
        """
        获取参数量统计

        Returns:
            dict 包含 total, density, backbone, trainable
        """
        total = sum(p.numel() for p in self.parameters())
        density = sum(p.numel() for p in self.density_net.parameters())
        backbone = sum(p.numel() for p in self.backbone.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            'total': total,
            'density': density,
            'backbone': backbone,
            'trainable': trainable,
        }

    def extra_repr(self) -> str:
        return (
            f"density_checkpoint={self.density_checkpoint}, "
            f"freeze_density={self.freeze_density}, "
            f"backbone_input_channels={self.backbone_input_channels}"
        )


def get_density_concat_model(
    density_checkpoint: str = "experiments/haze_density/checkpoints/formal/best.pth",
    freeze_density: bool = True,
    backbone_input_channels: int = 4,
    density_base_channels: int = 32,
) -> DensityConcatModel:
    """
    获取 DensityConcatModel 实例

    Args:
        density_checkpoint: HazeDensityNet checkpoint 路径
        freeze_density: 是否冻结 HazeDensityNet
        backbone_input_channels: Backbone 输入通道数
        density_base_channels: HazeDensityNet base_channels

    Returns:
        DensityConcatModel 实例
    """
    return DensityConcatModel(
        density_checkpoint=density_checkpoint,
        freeze_density=freeze_density,
        backbone_input_channels=backbone_input_channels,
        density_base_channels=density_base_channels,
    )


__all__ = ["DensityConcatModel", "get_density_concat_model"]
