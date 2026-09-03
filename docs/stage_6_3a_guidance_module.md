# Stage 6-3A: Density Attention Guidance Module

## 1. Overview

本阶段实现 **DensityGuidanceModule**，使用冻结的 HazeDensityNet 输出的 density map 对视觉 backbone feature 进行空间引导。

### 1.1 Design Goals

1. **Feature-level guidance**: 在 feature level 应用密度引导，不修改 backbone 的 3-channel RGB 输入
2. **Resolution alignment**: 自动对齐 density map [B,1,256,256] 与不同尺度 feature 的空间分辨率
3. **Identity initialization**: gamma=0 初始化，保证训练初期 output ≈ input
4. **Gradient isolation**: HazeDensityNet 冻结，不参与反向传播

### 1.2 Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    DensityGuidanceModule                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  visual_feature [B,C,H,W]  density_map [B,1,H0,W0]             │
│         │                        │                              │
│         │                        ▼                              │
│         │              ┌─────────────────────┐                  │
│         │              │ F.interpolate        │                  │
│         │              │ (bilinear)          │                  │
│         │              └───────────┬─────────┘                  │
│         │                          │                             │
│         │                          ▼                             │
│         │                 density_aligned                        │
│         │                 [B,1,H,W]                              │
│         │                          │                             │
│         │                          ▼                             │
│         │              ┌─────────────────────┐                  │
│         │              │ density_proj        │                  │
│         │              │ Conv2d(1,C,1×1)     │                  │
│         │              └───────────┬─────────┘                  │
│         │                          │                             │
│         ▼                          │                             │
│  ┌──────────────┐                  │                             │
│  │ visual_proj  │                  │                             │
│  │ Conv2d(C,C,1)│                  │                             │
│  └──────┬───────┘                  │                             │
│         │                          │                             │
│         ▼                          │                             │
│  visual_proj_out                   │                             │
│  [B,C,H,W]                         │                             │
│         │                          │                             │
│         └──────────┬───────────────┘                             │
│                    ▼                                             │
│              ┌─────────────┐                                     │
│              │   +         │                                     │
│              └──────┬──────┘                                     │
│                     │                                            │
│                     ▼                                            │
│              ┌─────────────┐                                     │
│              │  sigmoid    │                                     │
│              └──────┬──────┘                                     │
│                     │                                            │
│                     ▼                                            │
│            attention [B,C,H,W]                                   │
│                     │                                            │
│                     ▼                                            │
│  visual_feature + gamma * visual_feature * attention             │
│                     │                                            │
│                     ▼                                            │
│            guided_feature [B,C,H,W]                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 2. Mathematical Formulation

### 2.1 Spatial Alignment

给定 density map $D \in \mathbb{R}^{B \times 1 \times H_0 \times W_0}$ 和 visual feature $F \in \mathbb{R}^{B \times C \times H \times W}$，首先通过双线性插值对齐空间分辨率：

$$D_{\text{aligned}} = \text{Interpolate}(D, \text{size}=(H, W), \text{mode}=\text{bilinear})$$

### 2.2 Projection Layers

**Density Projection**:
$$D_{\text{proj}} = \text{Conv2d}_{1\times1}(D_{\text{aligned}}; W_d, b_d) \in \mathbb{R}^{B \times C \times H \times W}$$

其中 $W_d \in \mathbb{R}^{C \times 1 \times 1 \times 1}$。

**Visual Projection**:
$$F_{\text{proj}} = \text{Conv2d}_{1\times1}(F; W_v, b_v) \in \mathbb{R}^{B \times C \times H \times W}$$

其中 $W_v \in \mathbb{R}^{C \times C \times 1 \times 1}$。

### 2.3 Attention Map

$$A = \sigma(D_{\text{proj}} + F_{\text{proj}}) \in \mathbb{R}^{B \times C \times H \times W}$$

其中 $\sigma(\cdot)$ 为 sigmoid 激活函数，$A \in (0, 1)$。

### 2.4 Guided Feature

$$F_{\text{guided}} = F + \gamma \cdot (F \odot A)$$

其中：
- $\odot$ 表示 element-wise multiplication
- $\gamma \in \mathbb{R}$ 为可学习缩放因子
- 初始化 $\gamma = 0$，保证 $F_{\text{guided}} \approx F$

## 3. Tensor Shapes

### 3.1 Input/Output

| Tensor | Shape | Description |
|--------|-------|-------------|
| `visual_feature` | `[B, C, H, W]` | 视觉 backbone feature |
| `density_map` | `[B, 1, H0, W0]` | 密度图 (任意分辨率) |
| `guided_feature` | `[B, C, H, W]` | 引导后的 feature |

### 3.2 Intermediate Tensors

| Tensor | Shape | Description |
|--------|-------|-------------|
| `density_aligned` | `[B, 1, H, W]` | 对齐后的密度图 |
| `density_proj` | `[B, C, H, W]` | Density projection output |
| `visual_proj` | `[B, C, H, W]` | Visual projection output |
| `attention` | `[B, C, H, W]` | Attention map |

### 3.3 Multi-scale Configuration

针对 SimpleBackbone 的 4-stage 输出：

| Scale | Feature Shape | Density Map |
|-------|---------------|-------------|
| F0 | `[B, 128, 128, 128]` | `[B, 1, 256, 256]` |
| F1 | `[B, 256, 64, 64]` | `[B, 1, 256, 256]` |
| F2 | `[B, 512, 32, 32]` | `[B, 1, 256, 256]` |
| F3 | `[B, 1024, 16, 16]` | `[B, 1, 256, 256]` |

## 4. Parameter Analysis

### 4.1 Per-scale Parameters

对于 feature_channels = C：

| Component | Parameters | Formula |
|-----------|------------|---------|
| `density_proj` | C | $1 \times C \times 1 \times 1$ |
| `visual_proj` | C² | $C \times C \times 1 \times 1$ |
| `gamma` | 1 | scalar |
| **Total** | **C + C² + 1** | - |

### 4.2 Parameter Counts

| Scale | C | density_proj | visual_proj | gamma | Total |
|-------|---|--------------|-------------|-------|-------|
| F0 | 128 | 128 | 16,384 | 1 | 16,513 |
| F1 | 256 | 256 | 65,536 | 1 | 65,793 |
| F2 | 512 | 512 | 262,144 | 1 | 262,657 |
| F3 | 1024 | 1,024 | 1,048,576 | 1 | 1,049,601 |
| **All** | - | 1,920 | 1,392,640 | 4 | **1,394,564** |

### 4.3 Comparison with Stage 6-2

| Metric | Stage 6-2 (Concat) | Stage 6-3A (Guidance) |
|--------|-------------------|----------------------|
| Backbone input channels | 4 (RGB+D) | 3 (RGB only) |
| Additional parameters | ~13% backbone increase | 1.39M (guidance modules) |
| Memory overhead | Higher (4-channel features) | Lower (3-channel features) |
| Flexibility | Fixed at input | Feature-level, multi-scale |

## 5. Implementation Details

### 5.1 File Structure

```
src/models/density_guidance/
├── __init__.py              # Module exports
└── density_guidance.py      # DensityGuidanceModule implementation
```

### 5.2 Key Design Choices

1. **Bilinear interpolation**: 使用 `F.interpolate` 进行密度图下采样，保持空间语义
2. **1×1 convolutions**: 轻量级投影，不改变空间分辨率
3. **Sigmoid activation**: 将 attention 限制在 (0, 1) 范围
4. **Residual connection**: 保证信息不丢失，易于优化
5. **Gamma initialization**: $\gamma=0$ 确保初始 identity 行为

### 5.3 Code Snippets

**Module Initialization**:
```python
self.gamma = nn.Parameter(torch.zeros(1))  # gamma=0 for identity
```

**Spatial Alignment**:
```python
density_aligned = F.interpolate(
    density_map,
    size=visual_feature.shape[-2:],
    mode='bilinear',
    align_corners=False,
)
```

**Guided Feature**:
```python
guided_feature = visual_feature + self.gamma * visual_feature * attention
```

## 6. Test Results

### 6.1 Test Suite

| Test | Status | Description |
|------|--------|-------------|
| Shape Test | ⏳ Pending | F0/F1/F2/F3 shape consistency |
| Identity Init | ⏳ Pending | gamma=0 → output≈input |
| Gradient Flow | ⏳ Pending | visual_feature and params have grad |
| Density Sensitivity | ⏳ Pending | zeros vs ones density produce different output |
| No NaN/Inf | ⏳ Pending | Various input conditions |
| CUDA Test | ⏳ Pending | GPU forward for all scales |
| Parameter Stats | ⏳ Pending | Verify parameter counts |
| Multi-scale Integration | ⏳ Pending | Full multi-scale forward |

> **Note**: 测试脚本已创建 (`scripts/test_density_guidance.py`)，待运行验证。

### 6.2 Expected Results

**Shape Test**:
```
F0: [B,128,128,128] + [B,1,256,256] → [B,128,128,128] ✓
F1: [B,256,64,64] + [B,1,256,256] → [B,256,64,64] ✓
F2: [B,512,32,32] + [B,1,256,256] → [B,512,32,32] ✓
F3: [B,1024,16,16] + [B,1,256,256] → [B,1024,16,16] ✓
```

**Identity Initialization**:
```
gamma = 0.0
max(|guided - feature|) < 1e-5 ✓
```

**Gradient Flow**:
```
visual_feature.grad is not None ✓
module.density_proj.weight.grad is not None ✓
module.visual_proj.weight.grad is not None ✓
module.gamma.grad is not None ✓
```

## 7. Limitations & Future Work

### 7.1 Current Limitations

1. **Fixed attention mechanism**: 当前使用简单的 sigmoid fusion，可能不够 expressive
2. **Single gamma per scale**: 每个尺度共享一个 gamma，可能限制表达能力
3. **No channel-wise attention**: 当前 attention 是 channel-aware 但未显式建模 channel dependency

### 7.2 Potential Improvements

1. **SE-style attention**: 引入 channel-wise attention (Squeeze-and-Excitation)
2. **CBAM fusion**: 结合 channel 和 spatial attention
3. **Learnable interpolation**: 使用可学习的 downsampling 代替 bilinear
4. **Gating mechanism**: 引入更复杂的 gating (如 FiLM, Feature-wise Linear Modulation)

## 8. Integration with Main Task

### 8.1 Usage Pattern

```python
from src.models.density_guidance import create_density_guidance_modules

# 创建多尺度 guidance modules
guidance_modules = create_density_guidance_modules(
    feature_channels_list=(128, 256, 512, 1024)
)

# 在 backbone 后应用
features = backbone(image)  # List of [B,C,H,W]
guided_features = []
for module, feature in zip(guidance_modules, features):
    guided = module(feature, density_map)
    guided_features.append(guided)
```

### 8.2 Training Considerations

1. **Freeze HazeDensityNet**: `requires_grad=False`
2. **Warmup gamma**: 可选，逐渐增加 gamma 的学习率
3. **Loss weighting**: guidance module 可能影响 downstream task，需调整 loss weight

## 9. Acceptance Criteria

- [x] Module implemented
- [ ] F0 shape PASS
- [ ] F1 shape PASS
- [ ] F2 shape PASS
- [ ] F3 shape PASS
- [ ] Identity initialization PASS
- [ ] Gradient PASS
- [ ] Density sensitivity PASS
- [ ] NaN/Inf PASS
- [ ] CUDA PASS
- [ ] Parameters recorded
- [x] Documentation updated

> **Status**: Implementation complete, awaiting test execution.

## 10. References

- Stage 6-2 Report: `docs/stage_6-2_baseline_integration_report.md`
- HazeDensityNet: `src/models/haze_density/haze_density_net.py`
- SimpleBackbone: `src/models/backbone/simple_backbone.py`
