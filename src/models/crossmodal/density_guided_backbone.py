# -*- coding: utf-8 -*-
"""
Density Guided Backbone (Stage 6-3B)

将 DensityGuidanceModule 集成到 Backbone 的多尺度特征金字塔中。

架构:
    RGB [B, 3, H, W]
        ├──────────────→ HazeDensityNet (Frozen)
        │                     │
        │                     ↓
        │                  density [B, 1, H, W]
        │
        └──────────────→ Backbone (3-channel)
                            │
                        F0 F1 F2 F3
                         │  │  │  │
                         ↓  ↓  ↓  ↓
                        DG DG DG DG  (DensityGuidanceModule)
                         │  │  │  │
                         ↓  ↓  ↓  ↓
                        G0 G1 G2 G3  (Guided Features)

设计要点:
1. Backbone 保持 3-channel RGB 输入
2. HazeDensityNet 冻结，不参与反向传播
3. Density 在 feature level 进行 guidance
4. 支持外部传入 density (可选)

接口:
    guided_features, density = model(rgb_image, density=None)
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Tuple, List

from ..haze_density import HazeDensityNet
from ..backbone import SimpleBackbone
from ..density_guidance import create_density_guidance_modules


class DensityGuidedBackbone(nn.Module):
    """
    Density Guided Backbone

    Args:
        density_checkpoint: HazeDensityNet checkpoint 路径
        freeze_density: 是否冻结 HazeDensityNet (默认 True)
        density_base_channels: HazeDensityNet base_channels (默认 32)
    """

    def __init__(
        self,
        density_checkpoint: str = "experiments/haze_density/checkpoints/formal/best.pth",
        freeze_density: bool = True,
        density_base_channels: int = 32,
    ):
        super().__init__()

        self.density_checkpoint = density_checkpoint
        self.freeze_density = freeze_density

        # ========== HazeDensityNet (Frozen) ==========
        print(f"[DensityGuidedBackbone] Loading HazeDensityNet from {density_checkpoint}")
        self.density_net = HazeDensityNet(base_channels=density_base_channels, use_sigmoid=True)

        # 加载 checkpoint
        checkpoint_path = Path(density_checkpoint)
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            missing_keys, unexpected_keys = self.density_net.load_state_dict(
                checkpoint['model_state_dict'], strict=True
            )
            if missing_keys:
                raise ValueError(f"Missing keys in checkpoint: {missing_keys}")
            if unexpected_keys:
                raise ValueError(f"Unexpected keys in checkpoint: {unexpected_keys}")
            print(f"[DensityGuidedBackbone] Loaded density checkpoint successfully")
        else:
            raise FileNotFoundError(f"Density checkpoint not found: {density_checkpoint}")

        # 冻结密度网络
        if freeze_density:
            self.density_net.eval()
            for param in self.density_net.parameters():
                param.requires_grad = False
            print(f"[DensityGuidedBackbone] HazeDensityNet frozen (requires_grad=False)")
        else:
            self.density_net.train()
            print(f"[DensityGuidedBackbone] HazeDensityNet trainable (requires_grad=True)")

        # ========== Backbone (3-channel RGB input) ==========
        print(f"[DensityGuidedBackbone] Creating 3-channel backbone")
        self.backbone = SimpleBackbone(input_channels=3)

        # ========== Density Guidance Modules (4 scales) ==========
        print(f"[DensityGuidedBackbone] Creating 4-scale density guidance modules")
        self.guidance_modules = create_density_guidance_modules(
            feature_channels_list=(128, 256, 512, 1024)
        )

        # 统计参数量
        self._print_parameter_stats()

    def _print_parameter_stats(self):
        """打印参数量统计"""
        total_params = sum(p.numel() for p in self.parameters())
        density_params = sum(p.numel() for p in self.density_net.parameters())
        backbone_params = sum(p.numel() for p in self.backbone.parameters())
        guidance_params = sum(p.numel() for p in self.guidance_modules.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        print(f"\n[DensityGuidedBackbone] Parameter Statistics:")
        print(f"  Total parameters:      {total_params:,}")
        print(f"  HazeDensityNet:        {density_params:,} (frozen={self.freeze_density})")
        print(f"  Backbone:              {backbone_params:,}")
        print(f"  Guidance modules:      {guidance_params:,}")
        print(f"  Trainable parameters:  {trainable_params:,}")
        print()

    def forward(
        self,
        rgb_image: torch.Tensor,
        density: Optional[torch.Tensor] = None,
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        前向传播

        Args:
            rgb_image: RGB 图像 [B, 3, H, W], 范围 [0, 1]
            density: 可选的密度图 [B, 1, H, W]。如果为 None，则使用 HazeDensityNet 生成

        Returns:
            Tuple:
                - guided_features: List of guided features at 4 scales
                - density: 密度图 [B, 1, H, W]
        """
        # ========== 生成/获取密度图 ==========
        if density is None:
            if self.freeze_density:
                with torch.no_grad():
                    density = self.density_net(rgb_image)
            else:
                density = self.density_net(rgb_image)
        else:
            # 外部传入的 density，确保 detached
            density = density.detach() if self.freeze_density else density

        # ========== Backbone Forward ==========
        features = self.backbone(rgb_image)
        # features: [F0, F1, F2, F3]
        # F0: [B, 128, H/2, W/2]
        # F1: [B, 256, H/4, W/4]
        # F2: [B, 512, H/8, W/8]
        # F3: [B, 1024, H/16, W/16]

        # ========== Density Guidance ==========
        guided_features = []
        for i, (feature, guidance_module) in enumerate(zip(features, self.guidance_modules)):
            guided_feature = guidance_module(feature, density)
            guided_features.append(guided_feature)

        return guided_features, density

    def get_raw_features(
        self,
        rgb_image: torch.Tensor,
    ) -> List[torch.Tensor]:
        """
        获取未经 guidance 的原始 backbone 特征

        Args:
            rgb_image: RGB 图像 [B, 3, H, W]

        Returns:
            List of raw features at 4 scales
        """
        return self.backbone(rgb_image)

    def get_density(
        self,
        rgb_image: torch.Tensor,
    ) -> torch.Tensor:
        """
        仅获取密度图

        Args:
            rgb_image: RGB 图像 [B, 3, H, W]

        Returns:
            Density map [B, 1, H, W]
        """
        if self.freeze_density:
            with torch.no_grad():
                density = self.density_net(rgb_image)
        else:
            density = self.density_net(rgb_image)
        return density

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

        print(f"[DensityGuidedBackbone] Density network frozen={freeze}")

    def count_parameters(self) -> dict:
        """
        获取参数量统计

        Returns:
            dict 包含 total, density, backbone, guidance, trainable
        """
        total = sum(p.numel() for p in self.parameters())
        density = sum(p.numel() for p in self.density_net.parameters())
        backbone = sum(p.numel() for p in self.backbone.parameters())
        guidance = sum(p.numel() for p in self.guidance_modules.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable

        return {
            'total': total,
            'density': density,
            'backbone': backbone,
            'guidance': guidance,
            'trainable': trainable,
            'frozen': frozen,
        }

    def extra_repr(self) -> str:
        return (
            f"density_checkpoint={self.density_checkpoint}, "
            f"freeze_density={self.freeze_density}"
        )


def get_density_guided_backbone(
    density_checkpoint: str = "experiments/haze_density/checkpoints/formal/best.pth",
    freeze_density: bool = True,
    density_base_channels: int = 32,
) -> DensityGuidedBackbone:
    """
    获取 DensityGuidedBackbone 实例

    Args:
        density_checkpoint: HazeDensityNet checkpoint 路径
        freeze_density: 是否冻结 HazeDensityNet
        density_base_channels: HazeDensityNet base_channels

    Returns:
        DensityGuidedBackbone 实例
    """
    return DensityGuidedBackbone(
        density_checkpoint=density_checkpoint,
        freeze_density=freeze_density,
        density_base_channels=density_base_channels,
    )


__all__ = ["DensityGuidedBackbone", "get_density_guided_backbone"]
