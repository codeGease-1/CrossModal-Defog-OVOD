# Notebook 使用与迁移说明

更新于 2026-10-20。仓库目前有两份 notebook：

| 文件 | 定位 |
|---|---|
| `docs/雾密度网络_最简实验流程.ipynb` | 较短的环境、数据、划分、训练、可视化流程 |
| `docs/雾密度网络_更新后.ipynb` | 较长的历史实验与排错过程 |

本次审查前，两份文件已经存在本地改动/新增状态。本次保留它们，没有覆盖 cell、清除输出或执行历史修复命令。日常操作以 [最新操作指南](guides/haze_density_user_guide.md) 为准。

## 和当前代码配合时注意什么？

1. 确认 Colab 加载的是本次修正后的代码，`PRIOR_VERSION` 应为 `spatial_minmax_v2`。
2. 现有 notebook 可能仍使用 `formal/` 输出目录。新实验建议调用 `--config configs/haze_density.yaml`，并把后续可视化读取路径改为本次实际 checkpoint 路径。
3. 训练脚本新增防覆盖检查，同一个目录再次从头训练会拒绝；应恢复 `latest.pth` 或使用新目录。
4. 旧权重没有当前先验版本标记，不能直接继续为同一新实验；保留为历史资料。
5. Drive 持久保存输出的步骤见 [Colab 指南](guides/colab.md)，不要依赖 `/content` 的存活。
6. 不要重复运行历史 `fix_decoder_relu.py`、批量路径修复或 `git checkout` cell。当前代码已经修过相关问题，旧调试步骤不属于训练流程。
7. Notebook 中若内嵌了一份先验计算函数，应改用 `PhysicalPriorModule`，避免复制旧归一化错误。

## 路径规则

| 用途 | 当前入口 |
|---|---|
| 正式训练 | `scripts/train/train_haze_density.py` |
| 8 图过拟合排错 | `scripts/train/train_overfit_8.py` |
| 测试集评估 | `scripts/evaluate/evaluate_haze_density.py` |
| 预测分布审计 | `scripts/evaluate/audit_prediction_distribution.py` |
| 数据划分 | `scripts/data/generate_rshazeplus_split.py` |
| 数据与先验可视化 | `scripts/data/visualize_physical_prior_on_rshazeplus.py` |
| 静态检查 | `scripts/utils/static_check.py` |
| 模块检查 | `scripts/test/` |

`tools/update_notebook_paths.py` 是历史迁移工具，不会验证实验逻辑、标签版本、依赖或结果。需要迁移时先备份，明确传入和输出路径；不要把“替换了字符串”理解成“整个 notebook 已经验证能跑”。
