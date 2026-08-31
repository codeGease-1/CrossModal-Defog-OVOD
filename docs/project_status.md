# 项目状态跟踪

**项目名称**: CrossModal-Defog-OVOD - 跨模态语义恢复引导的含雾低质遥感影像开放词汇目标检测  
**当前模块**: 雾密度感知网络 (Haze Density Perception Network)  
**当前阶段**: Stage 5B-2: Dataset + Physical Prior Integration  
**最后更新**: 2026-08-31

---

## 一、开发模式

| 环境 | 职责 |
|------|------|
| **本地 (Windows)** | 代码开发、静态检查、Git 管理 |
| **Google Colab (T4 GPU)** | PyTorch 运行、模型训练、实验验证 |

---

## 二、阶段进度

### Phase 0: 工程初始化 [OK] 已完成

### Phase 1: 物理先验模块 [OK] 已完成

### Phase 2: 核心特征提取模块 [OK] 已完成

### Phase 3: 完整雾密度网络 [OK] 已完成（待 Colab 验证）

### Phase 4: 训练框架 [IN PROGRESS]

#### Stage 5A: 数据集调研与实验数据方案设计 [OK] 已完成

| 任务 | 状态 | 说明 |
|------|------|------|
| 数据集调研 | [OK] | RSHaze+ / RS-Haze / RRSHID |
| 实验方案设计 | [OK] | 主训练集 + 补充训练集 + 外部测试集 |
| Colab 数据目录规范 | [OK] |

#### Stage 5A.5: 数据集实际获取与核验 [OK] 已完成

| 任务 | 状态 | 说明 |
|------|------|------|
| RSHaze+ 下载 | [OK] | Zenodo |
| 目录结构验证 | [OK] | 7790 图像对 |
| 配对完整性 | [OK] | Hazy-Clear 配对 |

#### Stage 5B-1: RSHaze+ Dataset Implementation [OK] 已完成 (2026-08-31)

| 任务 | 状态 | 说明 |
|------|------|------|
| `RSHazePlusDataset` | [OK] | 核心 Dataset 类 |
| `build_rshazeplus_dataloader` | [OK] | DataLoader 构建器 |
| Split 系统 | [OK] | JSON-based, (subset, filename) key |
| Train/Val/Test | [OK] | 6174 / 686 / 930 |
| 256/512 模式 | [OK] | 双尺寸支持 |
| Colab 验证 | [OK] | 所有测试通过 |

**最终 Split**:
| Split | RSHaze_G | RSHaze_L | RSHaze_S | Total |
|-------|----------|----------|----------|-------|
| train | 900 | 4374 | 900 | **6174** |
| val | 100 | 486 | 100 | **686** |
| test | 330 | 270 | 330 | **930** |

#### Stage 5B-2: Dataset + Physical Prior Integration [IN PROGRESS]

| 任务 | 状态 | 说明 |
|------|------|------|
| Integration Test Script | [OK] | `test_rshazeplus_physical_prior.py` |
| Visualization Script | [OK] | `visualize_physical_prior_on_rshazeplus.py` |
| 256 forward test | [WAIT] | 【待 Colab】 |
| 512 forward test | [WAIT] | 【待 Colab】 |
| CUDA test | [WAIT] | 【待 Colab】 |
| G/L/S subset test | [WAIT] | 【待 Colab】 |
| Visualization | [WAIT] | 【待 Colab】 |
| Timing profile | [WAIT] | 【待 Colab】 |

**验收标准**:
- [ ] Dataset + Physical Prior 联调
- [ ] 256 forward PASS: [4,3,256,256] → [4,1,256,256]
- [ ] 512 forward PASS: [2,3,512,512] → [2,1,512,512]
- [ ] output range [0,1]
- [ ] finite = True
- [ ] CUDA PASS
- [ ] G/L/S subset PASS
- [ ] visualization PASS
- [ ] timing 完成

### Phase 5: Colab Smoke Test [WAIT] 待开始

| 任务 | 状态 | 说明 |
|------|------|------|
| 项目目录结构 | [OK] |
| `requirements.txt` | [OK] |
| `configs/haze_density.yaml` | [OK] |
| `docs/colab.md` | [OK] |
| `scripts/setup_colab.py` | [OK] |
| `scripts/static_check.py` | [OK] |

### Phase 1: 物理先验模块 [OK] 已完成

| 任务 | 状态 | 说明 |
|------|------|------|
| `physical_prior.py` | [OK] | Dark Channel / Local Contrast / Color Shift |
| `guided_filter.py` | [OK] | 引导滤波 |
| `scripts/test_physical_prior.py` | [OK] | 测试脚本 |

### Phase 2: 核心特征提取模块 [OK] 已完成

| 任务 | 状态 | 说明 |
|------|------|------|
| `basic_blocks.py` | [OK] | ConvBlock / DownsampleBlock |
| `encoder.py` | [OK] | Encoder |
| `residual_blocks.py` | [OK] | RB / SDRB |
| `eca.py` | [OK] | ECA |
| `multiscale.py` | [OK] | 多尺度分支 |
| `scripts/test_core_modules.py` | [OK] | 测试脚本 |

### Phase 3: 完整雾密度网络 [OK] 已完成（待 Colab 验证）

| 任务 | 状态 | 说明 |
|------|------|------|
| `fusion.py` | [OK] | 特征融合（Concat + Conv + ECA） |
| `decoder.py` | [OK] | Decoder 上采样 |
| `haze_density_net.py` | [OK] | 完整模型组装 |
| `scripts/test_model.py` | [OK] | 完整模型测试 |
| Shape Test | [WAIT] | 【在 Colab 执行】 |
| GPU Test | [WAIT] | 【在 Colab 执行】 |
| Full Pipeline | [WAIT] | 【在 Colab 执行】 |

### Phase 4: 训练框架 [WAIT] 待开始

| 任务 | 状态 | 说明 |
|------|------|------|
| `src/data/haze_dataset.py` | [WAIT] | Dataset 类 |
| `src/losses.py` | [WAIT] | MSE Loss |
| `src/train.py` | [WAIT] | 训练循环 |

### Phase 5: Colab Smoke Test [WAIT] 待开始

| 任务 | 状态 | 说明 |
|------|------|------|
| Colab 环境配置 | [WAIT] |
| 完整流程测试 | [WAIT] |

---

## 三、技术路线确认

### 申报书规定参数（不可修改）

| 参数 | 值 | 实现位置 |
|------|-----|----------|
| `weight_dark` | 0.5 | `physical_prior.py` |
| `weight_contrast` | 0.3 | `physical_prior.py` |
| `weight_color` | 0.2 | `physical_prior.py` |
| `exponent_mu` | 1.5 | `physical_prior.py` |
| `dilation_rates` | [2, 3, 4] | `multiscale.py` |

### 工程实现参数（可调整）

| 参数 | 当前值 | 说明 |
|------|--------|------|
| `base_channels` | 32 | Encoder 起始通道数 |
| `window_size` | 15 | 物理先验局部窗口 |
| `guided_radius` | 15 | 引导滤波半径 |
| `guided_eps` | 0.01 | 引导滤波正则化 |
| `use_sigmoid` | True | Decoder 输出激活（工程决策） |

---

## 四、下一步

### Stage 5B-2: Colab 验证

#### Step 1: 运行 Integration Test

```bash
# Colab T4
!pip install torch torchvision Pillow

# 运行测试
!python scripts/test_rshazeplus_physical_prior.py
```

**预期输出**:
```
Physical Prior Test (256x256)
Device: cuda
Train loader: 1544 batches
Batch 1: image [4,3,256,256] -> S_final [4,1,256,256], range=[..., ...], finite=True
...
[OK] 256 mode test passed

Physical Prior Test (512x512)
...
[OK] 512 mode test passed

Subset-Specific Test
...
[OK] All subsets tested

CUDA Test
...
[OK] CUDA test passed

[OK] 所有测试通过！
```

#### Step 2: 生成可视化

```bash
# 生成 Physical Prior 可视化
!python scripts/visualize_physical_prior_on_rshazeplus.py
```

**输出文件**:
- `experiments/haze_density/results/physical_prior/g_hazy_prior.png`
- `experiments/haze_density/results/physical_prior/l_hazy_prior.png`
- `experiments/haze_density/results/physical_prior/s_hazy_prior.png`
- `experiments/haze_density/results/physical_prior/sample_info.txt`

### 验收标准

Stage 5B-2 完成需满足:
- [ ] Dataset + Physical Prior 联调
- [ ] 256 forward PASS: [4,3,256,256] → [4,1,256,256]
- [ ] 512 forward PASS: [2,3,512,512] → [2,1,512,512]
- [ ] output range [0,1]
- [ ] finite = True
- [ ] CUDA PASS
- [ ] G/L/S subset PASS
- [ ] visualization PASS
- [ ] timing 完成

---

## 五、文件清单

### 新增文件 (Phase 3)

| 文件 | 说明 |
|------|------|
| `src/models/haze_density/fusion.py` | 特征融合模块 |
| `src/models/haze_density/decoder.py` | Decoder 上采样 |
| `src/models/haze_density/haze_density_net.py` | 完整模型 |
| `scripts/test_model.py` | 完整模型测试 |

### 新增文件 (Stage 5B-1)

| 文件 | 说明 |
|------|------|
| `src/data/datasets.py` | RSHazePlusDataset, HazeDensityDataset |
| `src/data/transforms.py` | HazeTrainTransform, HazeValTransform |
| `src/data/__init__.py` | 导出接口 + build_rshazeplus_dataloader |
| `scripts/generate_rshazeplus_split.py` | Split 生成脚本 |
| `scripts/test_rshazeplus_dataset.py` | Dataset 测试脚本 |
| `scripts/visualize_rshazeplus_dataset.py` | Dataset 可视化脚本 |
| `scripts/verify_rshazeplus_split.py` | Split 验证脚本 |
| `experiments/haze_density/rshazeplus_split.json` | 最终 split 文件 |

### 新增文件 (Stage 5B-2)

| 文件 | 说明 |
|------|------|
| `scripts/test_rshazeplus_physical_prior.py` | Dataset + Physical Prior 联调测试 |
| `scripts/visualize_physical_prior_on_rshazeplus.py` | Physical Prior 可视化 |

---

## 六、接口

### Dataset 接口

```python
from src.data import build_rshazeplus_dataloader, HazeDensityDataset

# DataLoader
train_loader = build_rshazeplus_dataloader(
    root='datasets/RSHaze+',
    split='train',
    image_size=256,
    batch_size=4,
    num_workers=2,
    split_file='experiments/haze_density/rshazeplus_split.json',
)

# Dataset
dataset = HazeDensityDataset(
    root='datasets/RSHaze+',
    split='train',
    image_size=256,
    return_clean=False,
    split_file='experiments/haze_density/rshazeplus_split.json',
)

sample = dataset[0]
# {
#     'image': tensor[3, H, W] [0, 1],
#     'subset': 'RSHaze_G/L/S',
#     'filename': '70.png',
#     'path': '/path/to/synhazypng/70.png',
# }
```

### Physical Prior 接口

```python
from src.models.haze_density.physical_prior import generate_s_final, PhysicalPriorModule
import torch

# Function API
image = torch.rand(4, 3, 256, 256)  # [B, 3, H, W] [0, 1]
s_final = generate_s_final(image)  # [B, 1, H, W] [0, 1]

# Module API
physical_prior = PhysicalPriorModule(
    window_size=15,
    guided_radius=15,
    guided_eps=0.01,
)
s_final = physical_prior(image)
```

### Dataset + Physical Prior 联调

```python
from src.data import build_rshazeplus_dataloader
from src.models.haze_density.physical_prior import PhysicalPriorModule
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# DataLoader
train_loader = build_rshazeplus_dataloader(
    root='datasets/RSHaze+',
    split='train',
    image_size=256,
    batch_size=4,
    split_file='experiments/haze_density/rshazeplus_split.json',
)

# Physical Prior
physical_prior = PhysicalPriorModule().to(device)
physical_prior.eval()

# Forward
for batch in train_loader:
    image = batch['image'].to(device)  # [4, 3, 256, 256]
    
    with torch.no_grad():
        s_final = physical_prior(image)  # [4, 1, 256, 256]
    
    print(f"S_final shape: {s_final.shape}")
    print(f"S_final range: [{s_final.min():.4f}, {s_final.max():.4f}]")
    print(f"S_final finite: {torch.isfinite(s_final).all()}")
    break
```

---

## 七、模型接口

```python
from src.models.haze_density import HazeDensityNet, generate_s_final
import torch

# 创建模型
model = HazeDensityNet(base_channels=32)

# 前向传播
image = torch.rand(2, 3, 256, 256)  # [B, 3, H, W]
pred = model(image)  # [B, 1, H, W]

# 训练时
target = generate_s_final(image)  # Physical-prior supervision target
loss = torch.nn.MSELoss()(pred, target)
loss.backward()
```

---

## 八、Stage 5B-2 注意事项

### 重要原则

1. **S_final 不是 Ground Truth**
   - S_final 是物理先验计算的监督信号 (physical-prior supervision target)
   - 不是人工标注的真实雾密度 ground truth

2. **训练目标**
   ```
   Prediction = HazeDensityNet(hazy_image)
   Target = PhysicalPrior(hazy_image)
   Loss = MSE(Prediction, Target)
   ```
   - 当前阶段不使用 clear image

3. **实时计算**
   - 当前采用实时计算 S_final
   - 暂不预先生成缓存文件
   - 后续根据 timing profile 决定是否缓存

---

**最后更新**: 2026-08-31 (Stage 5B-2 准备完成)
