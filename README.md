# CrossModal-Defog-OVOD

跨模态语义恢复引导的含雾低质遥感影像开放词汇目标检测

> **大学生创新训练项目** | 北京师范大学人工智能学院

---

## 项目概述

本项目旨在研究含雾低质遥感影像的开放词汇目标检测（Open-Vocabulary Object Detection, OVOD）方法。通过跨模态语义恢复技术，提升在雾霾等恶劣天气条件下的遥感影像解译能力。

### 当前阶段：雾密度感知网络

当前开发聚焦于**雾密度感知网络**模块，该模块用于生成精确的雾密度图，为后续的超分辨率和开放词汇分割提供基础。

---

## 开发模式

### 🖥️ 本地开发 + ☁️ Colab 执行

本项目采用**分离式开发模式**：

| 环境 | 职责 | 要求 |
|------|------|------|
| **本地 (Windows/Linux/Mac)** | 代码开发、Git 管理、静态检查 | 仅需 Python 环境（可选） |
| **Google Colab** | 模型训练、实验验证、GPU 推理 | NVIDIA GPU (T4 免费 tier) |

**重要**: 本地电脑**不需要**安装 PyTorch、CUDA 或任何 GPU 相关依赖。

---

## 项目结构

```
CrossModal-Defog-OVOD/
├── configs/                    # 配置文件
│   └── haze_density.yaml       # 雾密度网络配置
├── docs/                       # 文档
│   ├── colab.md                # Colab 使用说明
│   └── project_status.md       # 项目状态跟踪
├── experiments/                # 实验输出
│   └── haze_density/
│       ├── checkpoints/        # 模型检查点
│       └── logs/               # 训练日志
├── scripts/                    # 脚本
│   └── setup_colab.py          # Colab 环境设置
├── src/                        # 源代码
│   ├── data/                   # 数据模块
│   ├── models/                 # 模型模块
│   │   └── haze_density/       # 雾密度网络
│   └── utils/                  # 工具函数
├── requirements.txt            # Python 依赖
└── README.md                   # 本文件
```

---

## 快速开始

### 在 Google Colab 中运行

详见 [docs/colab.md](docs/colab.md)

简要步骤：

```python
# 1. 挂载 Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 2. 克隆项目
!git clone https://github.com/your-username/CrossModal-Defog-OVOD.git

# 3. 安装依赖
%cd /content/CrossModal-Defog-OVOD
!pip install -r requirements.txt

# 4. 运行训练
!python scripts/train_haze_density.py
```

---

## 技术路线

### 雾密度感知网络架构

```
输入：含雾 RGB 图像 I (B, 3, H, W)
    ↓
[物理先验分支]
    ↓ Dark Channel Prior (DCP)
    ↓ Local Contrast Prior (LCP)
    ↓ Color Shift Prior (CSP)
    ↓ 加权融合：S = 0.5×DCP + 0.3×LCP + 0.2×CSP
    ↓ 非线性变换：S_final = S^1.5
    ↓ Guided Filtering
    ↓ S_final (监督信号)

[深度网络分支]
    ↓ Encoder (下采样)
    ↓ 3 路并行多尺度 SDRB (dilation=2,3,4)
       → 每路：SDRB → RB → RB → ECA
    ↓ Concat + 3×3 Conv + ECA
    ↓ Decoder (上采样)
    ↓ 输出：I_h (B, 1, H, W)

Loss: MSE(I_h, S_final)
```

### 核心创新点

1. **物理先验耦合残差注意力**：暗通道/对比度/颜色偏移先验生成雾密度图
2. **多尺度残差分支**：dilation=2/3/4 捕获不同尺度雾密度特征
3. **ECA 通道注意力**：增强关键通道响应

---

## 配置说明

主要配置文件：`configs/haze_density.yaml`

### 申报书规定参数（不可修改）

```yaml
physical_prior:
  weight_dark: 0.5          # Dark Channel 权重
  weight_contrast: 0.3      # Local Contrast 权重
  weight_color: 0.2         # Color Shift 权重
  exponent_mu: 1.5          # 非线性指数
```

### 工程实现参数（可调整）

```yaml
data:
  image_size: 256           # 输入尺寸
  batch_size: 4             # 批量大小

model:
  base_channels: 32         # 基础通道数

train:
  lr: 1e-4                  # 学习率
  epochs: 100               # 训练轮数
```

---

## 依赖

见 [requirements.txt](requirements.txt)

主要依赖：
- PyTorch >= 2.0.0
- torchvision >= 0.15.0
- opencv-python >= 4.8.0
- numpy >= 1.24.0
- pyyaml >= 6.0

---

## 项目状态

当前进度详见 [docs/project_status.md](docs/project_status.md)

---

## 团队

**北京师范大学人工智能学院**  
跨模态语义恢复引导的含雾低质遥感影像开放词汇目标检测项目组

---

## 许可

本项目为大学生创新训练项目，仅供学术研究使用。
