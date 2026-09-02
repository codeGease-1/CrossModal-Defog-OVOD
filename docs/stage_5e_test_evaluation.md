# Stage 5E Report: HazeDensityNet Test Evaluation

**项目**: CrossModal-Defog-OVOD  
**阶段**: Stage 5E: Test Evaluation on RSHaze+  
**创建日期**: 2026-09-01  
**状态**: ⏸️ 待 Colab 执行

---

## 一、Checkpoint 信息

| 项目 | 值 |
|------|-----|
| **Checkpoint 路径** | `experiments/haze_density/checkpoints/formal/best.pth` |
| **Epoch** | 【待填】 |
| **Val Loss** | 【待填】 |
| **base_channels** | 32 |
| **use_sigmoid** | True |

---

## 二、Dataset 信息

### 2.1 Test Set 统计

| Subset | Samples |
|--------|---------|
| RSHaze_G | 330 |
| RSHaze_L | 270 |
| RSHaze_S | 330 |
| **Total** | **930** |

### 2.2 数据加载配置

| 参数 | 值 |
|------|-----|
| image_size | 256 |
| batch_size | 4 |
| num_workers | 0 |
| shuffle | False |
| transform | HazeValTransform (Resize + ToTensor) |

---

## 三、Overall Metrics

| 指标 | 值 |
|------|-----|
| **MSE** | 【待填】 |
| **MAE** | 【待填】 |
| **RMSE** | 【待填】 |
| **Pearson** | 【待填】 |

---

## 四、Metrics by Subset

### 4.1 RSHaze_G (330 samples)

| 指标 | 值 |
|------|-----|
| MAE | 【待填】 |
| RMSE | 【待填】 |
| Pearson | 【待填】 |

### 4.2 RSHaze_L (270 samples)

| 指标 | 值 |
|------|-----|
| MAE | 【待填】 |
| RMSE | 【待填】 |
| Pearson | 【待填】 |

### 4.3 RSHaze_S (330 samples)

| 指标 | 值 |
|------|-----|
| MAE | 【待填】 |
| RMSE | 【待填】 |
| Pearson | 【待填】 |

---

## 五、Prediction Distribution Audit

### 5.1 Prediction Distribution

| 指标 | 值 |
|------|-----|
| mean | 【待填】 |
| std | 【待填】 |
| min | 【待填】 |
| max | 【待填】 |
| p5 | 【待填】 |
| p25 | 【待填】 |
| p50 | 【待填】 |
| p75 | 【待填】 |
| p95 | 【待填】 |

### 5.2 Target Distribution

| 指标 | 值 |
|------|-----|
| mean | 【待填】 |
| std | 【待填】 |
| min | 【待填】 |
| max | 【待填】 |
| p5 | 【待填】 |
| p25 | 【待填】 |
| p50 | 【待填】 |
| p75 | 【待填】 |
| p95 | 【待填】 |

### 5.3 与 Validation Audit 对比

| 指标 | Validation | Test | 差异 |
|------|------------|------|------|
| mean | 【待填】 | 【待填】 | 【待填】 |
| std | 【待填】 | 【待填】 | 【待填】 |
| min | 【待填】 | 【待填】 | 【待填】 |
| max | 【待填】 | 【待填】 | 【待填】 |

---

## 六、Error Analysis

### 6.1 按密度区域误差

| 区域 | 像素占比 | MAE | RMSE |
|------|----------|-----|------|
| Low (<0.3) | 【待填】% | 【待填】 | 【待填】 |
| Medium (0.3-0.7) | 【待填】% | 【待填】 | 【待填】 |
| High (>=0.7) | 【待填】% | 【待填】 | 【待填】 |

### 6.2 按 Subset 误差对比

| Subset | MAE | RMSE | Pearson |
|--------|-----|------|---------|
| RSHaze_G | 【待填】 | 【待填】 | 【待填】 |
| RSHaze_L | 【待填】 | 【待填】 | 【待填】 |
| RSHaze_S | 【待填】 | 【待填】 | 【待填】 |
| **Overall** | 【待填】 | 【待填】 | 【待填】 |

---

## 七、可视化

### 7.1 输出路径

```
experiments/haze_density/results/test_evaluation/
├── RSHaze_G_*.png (16 samples)
├── RSHaze_L_*.png (16 samples)
├── RSHaze_S_*.png (16 samples)
└── test_metrics.txt
```

### 7.2 可视化格式

每个样本包含 4 列：
1. **Hazy Image**: 输入含雾图像
2. **Ground Truth**: Physical Prior S_final
3. **Prediction**: 模型预测雾密度
4. **Error Map**: 绝对误差 (归一化)

---

## 八、验收检查

| 检查项 | 状态 |
|--------|------|
| Test loader 正常 | 【待填】 |
| 无 NaN | 【待填】 |
| Prediction 范围 [0,1] | 【待填】 |
| 三个 subset 均完成 | 【待填】 |
| Metrics 保存 | 【待填】 |

---

## 九、Colab 执行命令

```bash
# 运行测试评估
!python scripts/evaluate_haze_density.py \
    --checkpoint experiments/haze_density/checkpoints/formal/best.pth \
    --image_size 256 \
    --batch_size 4 \
    --num_samples_per_subset 16

# 查看结果
!cat experiments/haze_density/results/test_evaluation/test_metrics.txt

# 列出可视化文件
!ls -la experiments/haze_density/results/test_evaluation/
```

---

## 十、结论

**待 Colab 执行后填写**

- 整体性能评估
- 各 subset 表现分析
- 与 validation 对比
- 潜在问题与建议

---

**报告生成日期**: 2026-09-01  
**作者**: 遥感智研助手
