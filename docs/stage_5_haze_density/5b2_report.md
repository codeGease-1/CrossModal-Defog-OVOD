# Stage 5B-2 Report: Dataset + Physical Prior Integration

**项目**: CrossModal-Defog-OVOD  
**阶段**: Stage 5B-2: RSHaze+ Dataset + Physical Prior Integration  
**创建日期**: 2026-08-31  
**状态**: ✅ 代码完成，待 Colab 验证

---

## 一、阶段目标

将 Stage 5B-1 完成的 RSHaze+ Dataset 与 Stage 2 完成的 Physical Prior 模块进行集成，验证完整数据流：

```
RSHazePlusDataset
    ↓
DataLoader
    ↓
image [B,3,H,W]
    ↓
Physical Prior
    ↓
S_final [B,1,H,W]
```

---

## 二、实现文件

| 文件 | 说明 | 状态 |
|------|------|------|
| `scripts/test_rshazeplus_physical_prior.py` | Integration 测试脚本 | ✅ 完成 |
| `scripts/visualize_physical_prior_on_rshazeplus.py` | Physical Prior 可视化 | ✅ 完成 |

---

## 三、Physical Prior 接口确认

### 3.1 主函数接口

```python
from src.models.haze_density.physical_prior import generate_s_final

# 输入：[B, 3, H, W] [0, 1]
image = torch.rand(4, 3, 256, 256)

# 输出：[B, 1, H, W] [0, 1]
s_final = generate_s_final(image)
```

### 3.2 Module 接口

```python
from src.models.haze_density.physical_prior import PhysicalPriorModule

physical_prior = PhysicalPriorModule(
    window_size=15,
    guided_radius=15,
    guided_eps=0.01,
)

s_final = physical_prior(image)  # [B, 1, H, W]
```

### 3.3 申报书规定参数

| 参数 | 值 | 位置 |
|------|-----|------|
| `weight_dark` | 0.5 | `physical_prior.py` |
| `weight_contrast` | 0.3 | `physical_prior.py` |
| `weight_color` | 0.2 | `physical_prior.py` |
| `exponent_mu` | 1.5 | `physical_prior.py` |

---

## 四、Colab 验证步骤

### 4.1 环境准备

```bash
# Colab T4
!pip install torch torchvision Pillow

# 检查 GPU
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
```

### 4.2 运行 Integration Test

```bash
python scripts/test_rshazeplus_physical_prior.py
```

**测试内容**:
1. 256 模式 forward: [4,3,256,256] → [4,1,256,256]
2. 512 模式 forward: [2,3,512,512] → [2,1,512,512]
3. G/L/S subset 分别测试
4. CUDA 设备测试
5. Timing profile

**预期输出**:
```
============================================================
Stage 5B-2: RSHaze+ Dataset + Physical Prior Integration Test
============================================================

============================================================
Physical Prior Test (256x256)
============================================================
Device: cuda
Train loader: 1544 batches
Batch 1: image [4,3,256,256] -> S_final [4,1,256,256], range=[0.0xxx, 0.9xxx], finite=True
...

Subset Statistics (S_final):
  RSHaze_G: count=xxx, mean=xxx, std=xxx, range=[xxx, xxx]
  RSHaze_L: count=xxx, mean=xxx, std=xxx, range=[xxx, xxx]
  RSHaze_S: count=xxx, mean=xxx, std=xxx, range=[xxx, xxx]

Timing Statistics (ms):
  Data loading: xx.xx ± xx.xx
  Physical prior: xx.xx ± xx.xx
  Total: xx.xx ± xx.xx

[OK] 256 mode test passed

...

[OK] 所有测试通过！
```

### 4.3 生成可视化

```bash
python scripts/visualize_physical_prior_on_rshazeplus.py
```

**输出文件**:
- `experiments/haze_density/results/physical_prior/g_hazy_prior.png` (RSHaze_G)
- `experiments/haze_density/results/physical_prior/l_hazy_prior.png` (RSHaze_L)
- `experiments/haze_density/results/physical_prior/s_hazy_prior.png` (RSHaze_S)
- `experiments/haze_density/results/physical_prior/sample_info.txt`

**可视化格式**:
```
┌─────────────┬─────────────┬─────────────┬─────────────┐
│    Hazy     │   S_final   │    Hazy     │   S_final   │
├─────────────┼─────────────┼─────────────┼─────────────┤
│    Hazy     │   S_final   │    Hazy     │   S_final   │
└─────────────┴─────────────┴─────────────┴─────────────┘
```

---

## 五、验收标准

| 验收项 | 预期值 | 状态 |
|--------|--------|------|
| Dataset + Physical Prior 联调 | PASS | ⏸️ 待 Colab |
| 256 forward | [4,3,256,256] → [4,1,256,256] | ⏸️ 待 Colab |
| 512 forward | [2,3,512,512] → [2,1,512,512] | ⏸️ 待 Colab |
| output shape | 正确 | ⏸️ 待 Colab |
| output range | [0, 1] | ⏸️ 待 Colab |
| finite | True | ⏸️ 待 Colab |
| CUDA | device=cuda | ⏸️ 待 Colab |
| RSHaze_G | PASS | ⏸️ 待 Colab |
| RSHaze_L | PASS | ⏸️ 待 Colab |
| RSHaze_S | PASS | ⏸️ 待 Colab |
| visualization | Generated | ⏸️ 待 Colab |
| timing | Profiled | ⏸️ 待 Colab |

---

## 六、重要原则

### 6.1 S_final 不是 Ground Truth

```
S_final = PhysicalPrior(hazy_image)
```

- S_final 是**物理先验计算的监督信号** (physical-prior supervision target)
- **不是**人工标注的真实雾密度 ground truth
- 用于监督 HazeDensityNet 学习雾密度估计

### 6.2 训练目标

```python
Prediction = HazeDensityNet(hazy_image)
Target = PhysicalPrior(hazy_image)  # S_final
Loss = MSE(Prediction, Target)
```

- 当前阶段**不使用** clear image
- clear image 将在后续阶段用于去雾任务

### 6.3 实时计算

- 当前采用**实时计算** S_final
- 暂不预先生成缓存文件
- 根据 timing profile 决定是否缓存

---

## 七、Expected Subset Statistics

不同 subset 的雾密度分布预期不同：

| Subset | 雾密度 | 预期 S_final mean |
|--------|--------|------------------|
| RSHaze_G | 一般雾 | 中等 |
| RSHaze_L | 轻雾 | 较低 |
| RSHaze_S | 浓雾 | 较高 |

**注意**: 不要因为数值分布不同就擅自修改 Physical Prior 参数。

---

## 八、下一步

1. 在 Colab T4 上执行测试脚本
2. 确认所有验收标准通过
3. 分析 timing profile，决定是否缓存 S_final
4. 更新 `docs/project_status.md`
5. 进入 Stage 5B-3: 训练循环实现

---

**报告生成日期**: 2026-08-31  
**作者**: 遥感智研助手
