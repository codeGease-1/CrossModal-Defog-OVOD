"""
CrossModal-Defog-OVOD - 跨模态语义恢复引导的含雾低质遥感影像开放词汇目标检测

项目结构:
    src/
    ├── data/           # 数据模块
    │   ├── datasets.py # Dataset 类
    │   └── transforms.py # 数据增强
    ├── models/         # 模型模块
    │   └── haze_density/  # 雾密度网络
    │       ├── basic_blocks.py
    │       ├── residual_blocks.py
    │       ├── eca.py
    │       ├── encoder.py
    │       ├── decoder.py
    │       └── haze_density_net.py
    └── utils/          # 通用工具预留目录

物理先验与引导滤波位于 src/models/haze_density/。

使用示例:
    from src.data import HazeDensityDataset
    from src.models.haze_density.haze_density_net import HazeDensityNet
    from src.models.haze_density.physical_prior import PhysicalPriorModule
"""

__version__ = "1.0.0"
__author__ = "CrossModal-Defog-OVOD Team"
__description__ = "跨模态语义恢复引导的含雾低质遥感影像开放词汇目标检测"
