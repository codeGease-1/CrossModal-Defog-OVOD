# 文档索引

更新于 2026-10-21。当前实现、运行命令和进度以本页列出的“当前文档”为准。

## 当前文档

| 文档 | 适合解决的问题 |
|---|---|
| [项目首页](../README.md) | 这个仓库目前能做什么？ |
| [雾密度入门解释](guides/haze_density_explained.md) | 原理是什么，模型到底在学什么？ |
| [操作指南](guides/haze_density_user_guide.md) | 数据怎么放，训练、恢复、评估怎么跑？ |
| [Colab 指南](guides/colab.md) | 环境和 Drive 持久化怎么配置？ |
| [项目状态](project_status.md) | 哪些有代码，哪些有本次验证证据？ |
| [阶段 2 科研审查](stage_2_research_review.md) | 独立真值评价发现了什么、哪些还未证明？ |
| [Benchmark 使用说明](guides/haze_density_benchmark.md) | 合成、排序、空间评价、性能和复现命令 |
| [独立 Colab notebook](haze_density_benchmark_colab.ipynb) | 在 VS Code 的 Colab 内核运行，无需远程获取本地代码 |
| [最简实验流程 notebook](雾密度网络_最简实验流程.ipynb) | 第 8 节在训练实验里直接运行独立 Benchmark |
| [阶段 2 方案](benchmark_plan.md) / [人工盲评协议](benchmark_human_protocol.md) | 评价定义、真实标签独立性与待开展盲评 |
| [阶段 1 必要修复报告](stage_1_bugfix_report.md) | 确认了哪些 Bug、实际测试结果、旧权重影响、阶段 2 前是否阻塞？ |
| [审查记录](project_review.md) | 发现了哪些错误，修了什么，还有什么限制？ |
| [Notebook 使用说明](notebook_update_guide.md) | 两个现有 notebook 如何与新流程配合？ |

## 历史阶段记录

以下保留当时的设计、排错和结果记录，包含待填项与未复核数值。历史文档的“已完成”不表示本次重新完成了该实验。特别是旧先验生成的指标，不可直接用于修正后的流程。

| 阶段 | 资料 |
|---|---|
| 数据实现 | [5B 数据实现](stage_5_haze_density/5b_dataset_implementation.md)、[5B-1 报告](stage_5_haze_density/5b1_report.md)、[5B-2 集成](stage_5_haze_density/5b2_report.md) |
| 过拟合与训练前检查 | [5C 8 图过拟合](stage_5_haze_density/5c_overfit_test.md)、[5D 训练前检查](stage_5_haze_density/5d_preflight.md) |
| 预测范围排错 | [5D-1 分布审计](stage_5_haze_density/5d1_prediction_audit.md)、[5D-2 Decoder 修复](stage_5_haze_density/5d2_decoder_verification.md) |
| 测试评估模板 | [5E 测试评估](stage_5_haze_density/5e_test_evaluation.md) |
| 集成原型 | [6-1 规划](stage_6_integration/6-1_integration_plan.md)、[6-2 拼接基线](stage_6_integration/6-2_baseline.md)、[6-3A 引导模块](stage_6_integration/6-3a_guidance_module.md)、[6-3B 集成](stage_6_integration/6-3b_guidance_integration.md) |

## 调研资料

[数据集调研](surveys/dataset_survey.md)、[数据获取清单](surveys/dataset_manifest.md)、[RSHaze+ 分析](surveys/rshazeplus_analysis.md)、[开放词汇分割可行性](surveys/ovs_feasibility.md)。这些是历史调研，下载方式、第三方模型和论文结论应在实际采用时再次核对。

## 日常入口

- 代码：`scripts/train/train_haze_density.py`、`scripts/evaluate/evaluate_haze_density.py`。
- 当前训练配置：`configs/haze_density.yaml`，使用 `--config` 读取。
- 实验文件：保存在各次运行指定的 `checkpoint_dir`、`result_dir` 和评估 `--output_dir`。
- 数据划分：`experiments/haze_density/rshazeplus_split.json`，随实验备份。
