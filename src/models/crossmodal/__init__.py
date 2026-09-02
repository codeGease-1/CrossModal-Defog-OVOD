"""
CrossModal 模块

包含跨模态语义恢复引导的含雾低质遥感影像开放词汇目标检测模型。

子模块:
    - density_concat_model: Density Concatenation Baseline (Stage 6-2)
    - density_guidance_model: Density Guidance Model (Stage 6-3, 待实现)
"""

from .density_concat_model import DensityConcatModel, get_density_concat_model

__all__ = [
    "DensityConcatModel",
    "get_density_concat_model",
]
