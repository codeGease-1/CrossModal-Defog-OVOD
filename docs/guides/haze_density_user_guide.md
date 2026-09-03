# 雾密度感知网络用户指南

**项目**: CrossModal-Defog-OVOD  
**模块**: HazeDensityNet (雾密度感知网络)  
**版本**: v1.0  
**最后更新**: 2026-09-03

---

## 目录

1. [快速开始](#1-快速开始)
2. [项目结构](#2-项目结构)
3. [环境配置](#3-环境配置)
4. [数据准备](#4-数据准备)
5. [模型训练](#5-模型训练)
6. [模型评估](#6-模型评估)
7. [可视化生成](#7-可视化生成)
8. [集成使用](#8-集成使用)
9. [常见问题](#9-常见问题)

---

## 1. 快速开始

### 1.1 最小可运行示例

```bash
# 训练 (5 epochs smoke test)
python scripts/train_haze_density.py --epochs 5

# 评估测试集
python scripts/evaluate_haze_density.py \
    --checkpoint experiments/haze_density/checkpoints/formal/best.pth

# 运行单元测试
python scripts/test_density_guidance.py
```

### 1.2 核心文件路径

| 文件 | 路径 |
|------|------|
| 训练脚本 | `scripts/train_haze_density.py` |
| 评估脚本 | `scripts/evaluate_haze_density.py` |
| 模型定义 | `src/models/haze_density/haze_density_net.py` |
| 数据集 | `src/data/datasets.py` |
| 默认 checkpoint | `experiments/haze_density/checkpoints/formal/best.pth` |

---

## 2. 项目结构

```
CrossModal-Defog-OVOD/
├── configs/                          # 配置文件
│   └── haze_density.yaml             # 雾密度网络配置
├── datasets/                         # 数据集目录
│   └── RSHaze+/                      # RSHaze+ 数据集
│       ├── RSHaze_G/
│       ├── RSHaze_L/
│       └── RSHaze_S/
├── docs/                             # 文档
│   ├── haze_density_user_guide.md    # 本文档
│   ├── stage_5d2_decoder_verification.md
│   └── stage_5e_test_evaluation.md
├── experiments/                       # 实验输出
│   └── haze_density/
│       ├── checkpoints/formal/       # 模型检查点
│       │   ├── best.pth              # 最佳模型
│       │   └── latest.pth            # 最新模型
│       ├── results/                  # 实验结果
│       │   ├── formal/              # 训练结果
│       │   └── test_evaluation/     # 测试评估
│       └── rshazeplus_split.json    # 数据划分
├── scripts/                          # 脚本
│   ├── train_haze_density.py        # 训练脚本
│   ├── evaluate_haze_density.py     # 评估脚本
│   ├── audit_prediction_distribution.py  # 分布审计
│   ├── test_density_guidance.py     # 单元测试
│   └── test_stage_6_3b.py           # 集成测试
├── src/                              # 源代码
│   ├── data/
│   │   ├── __init__.py
│   │   ├── datasets.py              # 数据集定义
│   │   └── transforms.py            # 数据变换
│   └── models/
│       ├── haze_density/            # 雾密度网络
│       │   ├── __init__.py
│       │   ├── haze_density_net.py  # 主模型
│       │   ├── encoder.py           # 编码器
│       │   ├── decoder.py           # 解码器
│       │   ├── multiscale.py        # 多尺度模块
│       │   ├── fusion.py            # 融合模块
│       │   ├── physical_prior.py    # 物理先验
│       │   └── guided_filter.py     # 导向滤波
│       ├── backbone/                # Backbone
│       ├── crossmodal/              # 跨模态模块
│       └── density_guidance/        # 密度引导
├── requirements.txt                  # 依赖
└── README.md                         # 项目说明
```

---

## 3. 环境配置

### 3.1 本地开发环境

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate     # Windows

# 安装依赖
pip install -r requirements.txt
```

### 3.2 Google Colab 环境

```python
# 1. 挂载 Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 2. 克隆项目
!git clone https://github.com/your-username/CrossModal-Defog-OVOD.git
%cd /content/CrossModal-Defog-OVOD

# 3. 安装依赖
!pip install -r requirements.txt

# 4. 验证安装
!python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

### 3.3 依赖列表

```txt
torch>=2.0.0
torchvision>=0.15.0
opencv-python>=4.8.0
numpy>=1.24.0
pyyaml>=6.0
Pillow>=9.0.0
tqdm>=4.65.0
```

---

## 4. 数据准备

### 4.1 RSHaze+ 数据集结构

```
datasets/RSHaze+/
├── RSHaze_G/
│   ├── train/
│   │   ├── cleanpng/      # 1000 张清晰图像
│   │   └── synhazypng/    # 1000 张含雾图像
│   └── test/
│       ├── cleanpng/      # 330 张
│       └── synhazypng/    # 330 张
├── RSHaze_L/
│   ├── train/
│   │   ├── cleanpng/      # ~2700 张
│   │   └── synhazypng/    # ~2700 张
│   └── test/
│       ├── cleanpng/      # 270 张
│       └── synhazypng/    # 270 张
└── RSHaze_S/
    ├── train/
    │   ├── cleanpng/      # 1000 张
    │   └── synhazypng/    # 1000 张
    └── test/
        ├── cleanpng/      # 330 张
        └── synhazypng/    # 330 张
```

### 4.2 数据划分

| Split | RSHaze_G | RSHaze_L | RSHaze_S | Total |
|-------|----------|----------|----------|-------|
| Train | 900 | 4374 | 900 | 6174 |
| Val | 100 | 486 | 100 | 686 |
| Test | 330 | 270 | 330 | 930 |

### 4.3 配置数据集路径

在 `configs/haze_density.yaml` 中修改:

```yaml
data:
  dataset_root: 'datasets/RSHaze+'
  split_file: 'experiments/haze_density/rshazeplus_split.json'
```

---

## 5. 模型训练

### 5.1 基本训练

```bash
# 5 epochs smoke test
python scripts/train_haze_density.py --epochs 5

# 完整训练 50 epochs
python scripts/train_haze_density.py --epochs 50
```

### 5.2 自定义参数训练

```bash
python scripts/train_haze_density.py \
    --epochs 100 \
    --batch_size 8 \
    --lr 5e-4 \
    --image_size 256 \
    --dataset_root datasets/RSHaze+
```

### 5.3 断点续训

```bash
# 从 latest checkpoint 继续
python scripts/train_haze_density.py \
    --resume experiments/haze_density/checkpoints/formal/latest.pth
```

### 5.4 训练输出

训练完成后生成以下文件:

```
experiments/haze_density/checkpoints/formal/
├── best.pth              # 最佳验证 loss 的模型
└── latest.pth            # 最后一个 epoch 的模型

experiments/haze_density/results/formal/
├── train_log.csv         # 训练日志 (每 epoch 指标)
├── training_summary.txt  # 训练总结
└── model_info.txt        # 模型信息
```

### 5.5 训练日志示例

```
Epoch  1/50: train_loss=0.023456, val_loss=0.018234, best=0.018234, time=12.3s
Epoch  2/50: train_loss=0.019876, val_loss=0.015432, best=0.015432, time=12.1s
...
```

---

## 6. 模型评估

### 6.1 测试集评估

```bash
python scripts/evaluate_haze_density.py \
    --checkpoint experiments/haze_density/checkpoints/formal/best.pth \
    --image_size 256 \
    --batch_size 4 \
    --num_samples_per_subset 16
```

### 6.2 评估指标

| 指标 | 说明 |
|------|------|
| MSE | 均方误差 |
| MAE | 平均绝对误差 |
| RMSE | 均方根误差 |
| Pearson | 皮尔逊相关系数 |

### 6.3 评估输出

```
experiments/haze_density/results/test_evaluation/
├── RSHaze_G_*.png        # RSHaze_G 可视化 (16 张)
├── RSHaze_L_*.png        # RSHaze_L 可视化 (16 张)
├── RSHaze_S_*.png        # RSHaze_S 可视化 (16 张)
└── test_metrics.txt      # 评估指标
```

### 6.4 查看评估结果

```bash
cat experiments/haze_density/results/test_evaluation/test_metrics.txt
```

输出示例:
```
Stage 5E: HazeDensityNet Test Evaluation Report
============================================================

Overall Metrics:
  MSE:     0.008234
  MAE:     0.071234
  RMSE:    0.090742
  Pearson: 0.876543

Metrics by Subset:

RSHaze_G:
  MAE:     0.068234
  RMSE:    0.088123
  Pearson: 0.881234

RSHaze_L:
  MAE:     0.072345
  RMSE:    0.091234
  Pearson: 0.875678

RSHaze_S:
  MAE:     0.073456
  RMSE:    0.092345
  Pearson: 0.872345
```

---

## 7. 可视化生成

### 7.1 预测分布审计

```bash
python scripts/audit_prediction_distribution.py \
    --checkpoint experiments/haze_density/checkpoints/formal/best.pth
```

输出:
```
experiments/haze_density/results/formal_prediction_audit/
├── audit_report.txt      # 审计报告
└── distribution_plot.png # 分布图
```

### 7.2 测试集可视化

评估脚本自动生成可视化，每个样本包含 4 列:

| 列 | 内容 |
|---|------|
| 1 | Hazy Image (输入含雾图像) |
| 2 | Ground Truth (Physical Prior S_final) |
| 3 | Prediction (模型预测雾密度) |
| 4 | Error Map (绝对误差，归一化) |

### 7.3 自定义可视化

```python
import torch
from PIL import Image
import numpy as np
from src.models.haze_density import HazeDensityNet
from src.models.haze_density.physical_prior import PhysicalPriorModule

# 加载模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = HazeDensityNet(base_channels=32, use_sigmoid=True)
model.load_state_dict(torch.load('experiments/haze_density/checkpoints/formal/best.pth')['model_state_dict'])
model.to(device)
model.eval()

# 加载物理先验
physical_prior = PhysicalPriorModule().to(device)
physical_prior.eval()

# 处理单张图像
image = Image.open('path/to/hazy_image.png').convert('RGB')
image_tensor = transforms(image).unsqueeze(0).to(device)

# 预测
with torch.no_grad():
    prediction = model(image_tensor)
    target = physical_prior(image_tensor)

# 可视化
def tensor_to_image(tensor):
    return tensor.squeeze().permute(1, 2, 0).cpu().numpy()

pred_img = tensor_to_image(prediction)
target_img = tensor_to_image(target)
error_img = np.abs(target_img - pred_img)
```

---

## 8. 集成使用

### 8.1 作为独立模块

```python
from src.models.haze_density import HazeDensityNet

# 创建模型
model = HazeDensityNet(base_channels=32, use_sigmoid=True)

# 加载 checkpoint
checkpoint = torch.load('experiments/haze_density/checkpoints/formal/best.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 推理
with torch.no_grad():
    density_map = model(hazy_image)  # [B, 1, H, W]
```

### 8.2 与 Backbone 集成 (Stage 6-3B)

```python
from src.models.crossmodal import DensityGuidedBackbone

# 创建集成模型
model = DensityGuidedBackbone(
    density_checkpoint='experiments/haze_density/checkpoints/formal/best.pth',
    freeze_density=True,
)

# 前向传播
guided_features, density = model(rgb_image)
# guided_features: [G0, G1, G2, G3] (4 尺度)
# density: [B, 1, H, W]
```

### 8.3 冻结密度网络

```python
# 设置冻结
model.set_freeze_density(True)

# 验证
for p in model.density_net.parameters():
    assert p.requires_grad == False
```

---

## 9. 常见问题

### 9.1 Checkpoint 加载失败

**问题**: `Missing keys in checkpoint`

**原因**: 代码版本与 checkpoint 不匹配

**解决**:
```bash
# 检查 checkpoint 信息
python -c "import torch; ckpt = torch.load('path/to/checkpoint.pth'); print(ckpt.keys())"

# 重新训练或使用匹配版本的代码
```

### 9.2 Prediction 范围异常

**问题**: Prediction 范围在 [0.5, 1) 而非 [0, 1]

**原因**: Decoder 中 `relu3 = nn.ReLU()` 导致 ReLU → Sigmoid 范围限制

**解决**: 已修复，确认 `decoder.py` 第 95 行为:
```python
self.relu3 = nn.Identity()  # 不是 nn.ReLU
```

### 9.3 CUDA Out of Memory

**问题**: GPU 显存不足

**解决**:
```bash
# 减小 batch size
python scripts/train_haze_density.py --batch_size 2

# 或使用 CPU (慢)
python scripts/train_haze_density.py --no_amp
```

### 9.4 数据集路径错误

**问题**: `FileNotFoundError: datasets/RSHaze+`

**解决**:
```bash
# 检查数据集目录
ls -la datasets/RSHaze+/

# 或修改配置
python scripts/train_haze_density.py --dataset_root /path/to/your/dataset
```

### 9.5 梯度消失/爆炸

**问题**: Loss 为 NaN 或 Inf

**解决**:
```bash
# 检查学习率
python scripts/train_haze_density.py --lr 1e-4  # 降低学习率

# 使用 AMP 混合精度
python scripts/train_haze_density.py --amp
```

---

## 附录 A: 模型架构

### A.1 HazeDensityNet 结构

```
输入：含雾 RGB 图像 [B, 3, H, W]
    ↓
Encoder (下采样 2x) → [B, 64, H/2, W/2]
    ↓
MultiScale (3 路并行，dilation=2/3/4)
    ↓
Fusion (Concat + Conv + ECA) → [B, 64, H/2, W/2]
    ↓
Decoder (上采样 2x) → [B, 1, H, W]
    ↓
Sigmoid → 雾密度图 [B, 1, H, W]
```

### A.2 Physical Prior 计算

```
Dark Channel Prior (DCP): 0.5 权重
Local Contrast Prior (LCP): 0.3 权重
Color Shift Prior (CSP): 0.2 权重
    ↓
加权融合：S = 0.5×DCP + 0.3×LCP + 0.2×CSP
    ↓
非线性变换：S_final = S^1.5
    ↓
Guided Filtering (平滑)
    ↓
监督信号 S_final [B, 1, H, W]
```

---

## 附录 B: 参数量统计

| 组件 | 参数量 | 占比 |
|------|--------|------|
| Encoder | ~100K | 20% |
| MultiScale | ~150K | 30% |
| Fusion | ~50K | 10% |
| Decoder | ~200K | 40% |
| **总计** | **~500K** | **100%** |

---

## 附录 C: 训练超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| base_channels | 32 | 基础通道数 |
| batch_size | 4 | 批量大小 |
| image_size | 256 | 输入尺寸 |
| lr | 1e-4 | 学习率 |
| epochs | 50 | 训练轮数 |
| weight_decay | 0 | L2 正则化 |
| amp | True | 混合精度训练 |

---

**文档维护**: 遥感智研助手  
**更新日期**: 2026-09-03  
**版本**: v1.0
