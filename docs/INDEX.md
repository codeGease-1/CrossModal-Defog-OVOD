# 文档索引

**项目**: CrossModal-Defog-OVOD  
**最后更新**: 2026-09-03

---

## 📚 用户指南

| 文档 | 说明 |
|------|------|
| [haze_density_user_guide.md](guides/haze_density_user_guide.md) | 雾密度感知网络完整使用指南 |
| [colab.md](guides/colab.md) | Google Colab 使用说明 |

---

## 📊 Stage 5: 雾密度感知网络

| 文档 | 说明 |
|------|------|
| [5b1_report.md](stage_5_haze_density/5b1_report.md) | Stage 5B-1 数据集实现报告 |
| [5b2_report.md](stage_5_haze_density/5b2_report.md) | Stage 5B-2 数据集与物理先验集成 |
| [5c_overfit_test.md](stage_5_haze_density/5c_overfit_test.md) | Stage 5C 8 图像过拟合测试 |
| [5d_preflight.md](stage_5_haze_density/5d_preflight.md) | Stage 5D-0 训练前审计 |
| [5d1_prediction_audit.md](stage_5_haze_density/5d1_prediction_audit.md) | Stage 5D-1 预测分布审计 |
| [5d2_decoder_verification.md](stage_5_haze_density/5d2_decoder_verification.md) | Stage 5D-2 Decoder 验证 |
| [5e_test_evaluation.md](stage_5_haze_density/5e_test_evaluation.md) | Stage 5E 测试集评估 |

---

## 🔗 Stage 6: 集成模块

| 文档 | 说明 |
|------|------|
| [6-1_integration_plan.md](stage_6_integration/6-1_integration_plan.md) | 集成方案规划 |
| [6-2_baseline.md](stage_6_integration/6-2_baseline.md) | Baseline 集成 |
| [6-3a_guidance_module.md](stage_6_integration/6-3a_guidance_module.md) | 密度引导模块 |
| [6-3b_guidance_integration.md](stage_6_integration/6-3b_guidance_integration.md) | 引导集成验证 |

---

## 📋 调研报告

| 文档 | 说明 |
|------|------|
| [rshazeplus_analysis.md](surveys/rshazeplus_analysis.md) | RSHaze+ 数据集分析 |
| [ovs_feasibility.md](surveys/ovs_feasibility.md) | 直接含雾 OVS 路径可行性评估 |

---

## 📝 项目状态

| 文档 | 说明 |
|------|------|
| [project_status.md](project_status.md) | 项目进度跟踪 |

---

## 🚀 快速导航

### 新手入门
1. 阅读 [haze_density_user_guide.md](guides/haze_density_user_guide.md)
2. 配置环境 ([colab.md](guides/colab.md))
3. 运行训练脚本

### 查看实验结果
- 训练日志：`experiments/haze_density/results/formal/train_log.csv`
- 测试指标：`experiments/haze_density/results/test_evaluation/test_metrics.txt`
- 可视化：`experiments/haze_density/results/test_evaluation/`

### 脚本使用
- 训练：`scripts/train/train_haze_density.py`
- 评估：`scripts/evaluate/evaluate_haze_density.py`
- 测试：`scripts/test/test_*.py`

---

**文档维护**: 遥感智研助手
