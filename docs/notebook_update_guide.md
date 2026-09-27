# Notebook 更新指南

**适用版本**: 雾密度网络.ipynb  
**原因**: 项目文件夹整理后路径变更  
**自动生成**: 使用 `tools/update_notebook_paths.py` 脚本批量更新  
**更新后文件**: `docs/雾密度网络_更新后.ipynb`

---

## ✅ 已完成的路径自动替换

以下路径已全部自动替换完成:

### scripts/ 路径变更

| 旧路径 | 新路径 | 状态 |
|--------|--------|------|
| `scripts/train_haze_density.py` | `scripts/train/train_haze_density.py` | ✅ |
| `scripts/train_overfit_8.py` | `scripts/train/train_overfit_8.py` | ✅ |
| `scripts/evaluate_haze_density.py` | `scripts/evaluate/evaluate_haze_density.py` | ✅ |
| `scripts/audit_prediction_distribution.py` | `scripts/evaluate/audit_prediction_distribution.py` | ✅ |
| `scripts/smoke_test.py` | `scripts/utils/smoke_test.py` | ✅ |
| `scripts/test_physical_prior.py` | `scripts/test/test_physical_prior.py` | ✅ |
| `scripts/visualize_physical_prior.py` | `scripts/data/visualize_physical_prior.py` | ✅ |
| `scripts/test_core_modules.py` | `scripts/test/test_core_modules.py` | ✅ |
| `scripts/test_model.py` | `scripts/test/test_model.py` | ✅ |
| `scripts/inspect_dataset.py` | `scripts/data/inspect_dataset.py` | ✅ |
| `scripts/generate_rshazeplus_split.py` | `scripts/data/generate_rshazeplus_split.py` | ✅ |
| `scripts/test_rshazeplus_dataset.py` | `scripts/test/test_rshazeplus_dataset.py` | ✅ |
| `scripts/verify_rshazeplus_split.py` | `scripts/data/verify_rshazeplus_split.py` | ✅ |
| `scripts/visualize_rshazeplus_dataset.py` | `scripts/data/visualize_rshazeplus_dataset.py` | ✅ |
| `scripts/test_rshazeplus_physical_prior.py` | `scripts/test/test_rshazeplus_physical_prior.py` | ✅ |
| `scripts/visualize_physical_prior_on_rshazeplus.py` | `scripts/data/visualize_physical_prior_on_rshazeplus.py` | ✅ |
| `scripts/fix_decoder_relu.py` | `scripts/utils/fix_decoder_relu.py` | ✅ |
| `scripts/test_density_concat_integration.py` | `scripts/test/test_density_concat_integration.py` | ✅ |
| `scripts/test_density_guidance.py` | `scripts/test/test_density_guidance.py` | ✅ |
| `scripts/test_stage_6_3b.py` | `scripts/test/test_stage_6_3b.py` | ✅ |

### docs/ 路径变更

| 旧路径 | 新路径 |
|--------|--------|
| `docs/stage_5b2_report.md` | `docs/stage_5_haze_density/5b2_report.md` |
| `docs/stage_5c_overfit_report.md` | `docs/stage_5_haze_density/5c_overfit_test.md` |
| `docs/stage_5d1_prediction_audit_report.md` | `docs/stage_5_haze_density/5d1_prediction_audit.md` |
| `docs/stage_5d2_decoder_verification.md` | `docs/stage_5_haze_density/5d2_decoder_verification.md` |
| `docs/stage_6-2_baseline_integration_report.md` | `docs/stage_6_integration/6-2_baseline.md` |
| `docs/stage_6_3a_guidance_module.md` | `docs/stage_6_integration/6-3a_guidance_module.md` |
| `docs/stage_6_3b_guidance_integration.md` | `docs/stage_6_integration/6-3b_guidance_integration.md` |

---

## 🗑️ 可删除的 cell 清单

根据对原 notebook 的分析，以下类型的 cell 建议删除:

### 1. Deprecation Warning 输出 cell（高优先级）

这些 cell 仅包含 PyTorch/Pillow 版本的 deprecation 警告，无实际保留价值:

| 警告类型 | 识别特征 | 建议删除 |
|----------|----------|---------|
| GradScaler Warning | `FutureWarning: torch.cuda.amp.GradScaler` | ✅ |
| autocast Warning | `FutureWarning: torch.cuda.amp.autocast` | ✅ |
| Pillow mode Warning | `DeprecationWarning: 'mode' parameter is deprecated` | ✅ |

### 2. 已修复问题的调试记录（中优先级）

| 主题 | 内容描述 | 状态 |
|------|----------|------|
| `fix_decoder_relu.py` 调用 | Decoder RELU 激活函数修复过程 | ✅ 已修复，可删除 |
| `!git checkout -- scripts/...` | 回退代码的 git 命令 | ✅ 历史操作，可删除 |
| 首次运行 `FileNotFoundError` | `No such file or directory` 错误 | ✅ 可删除 |

### 3. 过时脚本引用（已自动替换）

以下脚本调用已自动更新路径，cell 可保留:

- `scripts/test_core_modules.py` → `scripts/test/test_core_modules.py`
- `scripts/inspect_dataset.py` → `scripts/data/inspect_dataset.py`

---

## 自动更新报告

运行 `python tools/update_notebook_paths.py` 生成的报告:

```
Total cells: 61
Path replacements made: 31
Cells marked for deletion: 2
```

### 可删除 Cell 详情

| Cell # | 类型 | 内容预览 |
|--------|------|----------|
| 10 | git checkout revert | `!git checkout -- scripts/train/train_haze_density.py` |
| 42 | fix_decoder_relu | `!python scripts/utils/fix_decoder_relu.py` |

---

## 建议使用的新工作流程

### 训练

```python
# 基础训练 (5 epochs smoke test)
!python scripts/train/train_haze_density.py --epochs 5

# 完整训练
!python scripts/train/train_haze_density.py --epochs 50

# 8 图像过拟合
!python scripts/train/train_overfit_8.py
```

### 评估

```python
# 测试集评估
!python scripts/evaluate/evaluate_haze_density.py

# 预测分布审计
!python scripts/evaluate/audit_prediction_distribution.py --checkpoint experiments/haze_density/checkpoints/formal/best.pth
```

### 测试

```python
# 密度引导模块测试
!python scripts/test/test_density_guidance.py

# Stage 6-3B 集成测试
!python scripts/test/test_stage_6_3b.py

# 密度集成测试
!python scripts/test/test_density_concat_integration.py
```

### 数据

```python
# 生成 split 文件
!python scripts/data/generate_rshazeplus_split.py

# 验证 split
!python scripts/data/verify_rshazeplus_split.py

# 可视化物理先验
!python scripts/data/visualize_physical_prior_on_rshazeplus.py
```

---

## 批量替换命令 (Colab 中执行)

```python
# 克隆最新代码后执行
!cd /content/CrossModal-Defog-OVOD
!sed -i 's|scripts/train_haze_density.py|scripts/train/train_haze_density.py|g' 雾密度网络.ipynb
!sed -i 's|scripts/train_overfit_8.py|scripts/train/train_overfit_8.py|g' 雾密度网络.ipynb
!sed -i 's|scripts/evaluate_haze_density.py|scripts/evaluate/evaluate_haze_density.py|g' 雾密度网络.ipynb
!sed -i 's|scripts/audit_prediction_distribution.py|scripts/evaluate/audit_prediction_distribution.py|g' 雾密度网络.ipynb
!sed -i 's|scripts/test_physical_prior.py|scripts/test/test_physical_prior.py|g' 雾密度网络.ipynb
!sed -i 's|scripts/visualize_physical_prior.py|scripts/data/visualize_physical_prior.py|g' 雾密度网络.ipynb
!sed -i 's|scripts/test_core_modules.py|scripts/test/test_core_modules.py|g' 雾密度网络.ipynb
!sed -i 's|scripts/test_model.py|scripts/test/test_model.py|g' 雾密度网络.ipynb
!sed -i 's|scripts/generate_rshazeplus_split.py|scripts/data/generate_rshazeplus_split.py|g' 雾密度网络.ipynb
!sed -i 's|scripts/verify_rshazeplus_split.py|scripts/data/verify_rshazeplus_split.py|g' 雾密度网络.ipynb
!sed -i 's|scripts/inspect_dataset.py|scripts/data/inspect_dataset.py|g' 雾密度网络.ipynb
```

---

## 验证替换

替换后运行以下 cell 验证:

```python
# 验证脚本可访问
!ls -la scripts/train/
!ls -la scripts/evaluate/
!ls -la scripts/test/
!ls -la scripts/data/
!ls -la scripts/utils/

# 验证文档可访问
!ls -la docs/stage_5_haze_density/
!ls -la docs/stage_6_integration/
!ls -la docs/guides/
```

---

**更新日期**: 2026-09-03  
**维护者**: 遥感智研助手