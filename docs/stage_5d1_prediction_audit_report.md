# Stage 5D-1 Report: Prediction Distribution Audit

**项目**: CrossModal-Defog-OVOD  
**阶段**: Stage 5D-1: Prediction Distribution Audit  
**创建日期**: 2026-08-31  
**状态**: ⏸️ 待 Colab 验证

---

## 一、问题描述

在 Stage 5D 的 5-epoch smoke training 中发现：

| 指标 | 值 |
|------|-----|
| **Prediction min** | 0.5000 |
| **Prediction max** | 0.9291 |
| **Target min** | 0.0196 |
| **Target max** | 0.9784 |

**异常**: Prediction 最小值为 0.5，无法覆盖 Target 的低密度区域 [0, 0.5)。

---

## 二、问题根源分析

### 2.1 Decoder 结构检查

当前 `decoder.py` 第 86-124 行：

```python
# Conv 2 (输出 1 通道)
self.conv2 = nn.Conv2d(
    in_channels // 4,
    1,
    kernel_size=3,
    padding=1,
    bias=False,
)
self.norm3 = nn.InstanceNorm2d(1)
self.relu3 = nn.ReLU(inplace=True)  # ← 问题所在

# Sigmoid activation
self.sigmoid = nn.Sigmoid() if use_sigmoid else nn.Identity()

def forward(self, x: torch.Tensor) -> torch.Tensor:
    ...
    x = self.conv2(x)  # [B, 1, H, W]
    x = self.norm3(x)
    x = self.relu3(x)  # ReLU 输出 [0, ∞)
    
    # Sigmoid (工程实现决策)
    if self.use_sigmoid:
        x = self.sigmoid(x)  # Sigmoid([0, ∞)) = [0.5, 1)
    
    return x
```

### 2.2 数学分析

```
ReLU(x) = max(0, x)
         输出范围：[0, ∞)

Sigmoid(ReLU(x)) = Sigmoid([0, ∞))
                  = [Sigmoid(0), Sigmoid(∞))
                  = [0.5, 1)
```

**结论**: `ReLU → Sigmoid` 的组合导致输出范围被限制在 `[0.5, 1)`，无法预测低于 0.5 的雾密度值。

### 2.3 与申报书对比

申报书 3.2.1 节规定输出雾密度图 `I_h` 范围应为 `[0, 1]`。

当前实现：
- ❌ 输出范围 `[0.5, 1)`，与申报书冲突
- ❌ 无法预测低雾密度区域
- ❌ 导致训练 loss 无法有效下降

---

## 三、完整统计（待 Colab 验证）

### 3.1 Prediction Distribution

| 指标 | 值 |
|------|-----|
| min | 【待填】 |
| max | 【待填】 |
| mean | 【待填】 |
| std | 【待填】 |
| p1 | 【待填】 |
| p5 | 【待填】 |
| p25 | 【待填】 |
| p50 | 【待填】 |
| p75 | 【待填】 |
| p95 | 【待填】 |
| p99 | 【待填】 |

### 3.2 Target Distribution

| 指标 | 值 |
|------|-----|
| min | 【待填】 |
| max | 【待填】 |
| mean | 【待填】 |
| std | 【待填】 |
| p1 | 【待填】 |
| p5 | 【待填】 |
| p25 | 【待填】 |
| p50 | 【待填】 |
| p75 | 【待填】 |
| p95 | 【待填】 |
| p99 | 【待填】 |

### 3.3 低于阈值像素比例

| 阈值 | Prediction | Target |
|------|------------|--------|
| <0.1 | 【待填】% | 【待填】% |
| <0.2 | 【待填】% | 【待填】% |
| <0.3 | 【待填】% | 【待填】% |
| <0.4 | 【待填】% | 【待填】% |
| <0.5 | 【待填】% | 【待填】% |

### 3.4 误差指标

| 指标 | 值 |
|------|-----|
| Pearson correlation | 【待填】 |
| MAE | 【待填】 |
| RMSE | 【待填】 |
| MSE | 【待填】 |

### 3.5 按密度区域误差

| 区域 | 像素占比 | MAE | RMSE |
|------|----------|-----|------|
| Low (<0.3) | 【待填】% | 【待填】 | 【待填】 |
| Medium (0.3-0.7) | 【待填】% | 【待填】 | 【待填】 |
| High (>=0.7) | 【待填】% | 【待填】 | 【待填】 |

---

## 四、修复方案

### 4.1 最小修改方案

修改 `decoder.py` 第 94 行和第 120 行：

**方案 A（推荐）**: 移除最后的 ReLU

```python
# 原代码
self.relu3 = nn.ReLU(inplace=True)
...
x = self.relu3(x)

# 修改为
self.relu3 = nn.Identity()  # 或完全删除
...
# 删除 x = self.relu3(x)
```

**方案 B**: 保留 ReLU 但移除 Sigmoid

```python
# 不推荐，因为输出可能超出 [0, 1]
```

### 4.2 修复后预期

| 指标 | 修复前 | 修复后预期 |
|------|--------|------------|
| Prediction min | 0.5000 | ~0.0 |
| Prediction max | 0.9291 | ~1.0 |
| 输出范围 | [0.5, 1) | [0, 1] |

---

## 五、Colab 执行命令

```bash
# 运行 audit 脚本
!python scripts/audit_prediction_distribution.py \
    --checkpoint experiments/haze_density/checkpoints/formal/best.pth
```

---

## 六、结论

**预期结论**: C

- Prediction distribution 明显异常
- 原因：Decoder 中存在 ReLU → Sigmoid 结构
- 影响：Prediction 全部 >= 0.5，无法预测低雾密度区域
- 建议：移除 Decoder 最后的 ReLU
- 修复后需要重新进行 5 epoch smoke training

---

**报告生成日期**: 2026-08-31  
**作者**: 遥感智研助手
