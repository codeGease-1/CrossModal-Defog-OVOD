"""
Backbone 模块

包含用于特征提取的 Backbone 模型。

子模块:
    - simple_backbone: 简化 Backbone (用于 Baseline Testing)
"""

from .simple_backbone import SimpleBackbone, get_simple_backbone, init_4channel_from_3channel

__all__ = [
    "SimpleBackbone",
    "get_simple_backbone",
    "init_4channel_from_3channel",
]
