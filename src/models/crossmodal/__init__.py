"""
CrossModal 模块

包含跨模态语义恢复引导的含雾低质遥感影像开放词汇目标检测模型。

子模块:
    - density_concat_model: Density Concatenation Baseline (Stage 6-2)
    - density_guided_backbone: Density Guided Backbone (Stage 6-3)
"""

from .density_concat_model import DensityConcatModel, get_density_concat_model
from .density_guided_backbone import DensityGuidedBackbone, get_density_guided_backbone

__all__ = [
    "DensityConcatModel",
    "get_density_concat_model",
    "DensityGuidedBackbone",
    "get_density_guided_backbone",
]
