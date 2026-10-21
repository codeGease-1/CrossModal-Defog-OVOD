# Notebook 使用与迁移说明

更新于 2026-10-21。仓库目前有三份 notebook：

| 文件 | 定位 |
|---|---|
| `docs/雾密度网络_最简实验流程.ipynb` | 已同步 `codex-v1` / v3：环境检查、Drive 持久化、固定配置与划分、训练恢复、验证集看图、最终测试和第 8 节独立 Benchmark |
| `docs/雾密度网络_更新后.ipynb` | 较长的历史实验与排错过程 |
| `docs/haze_density_benchmark_colab.ipynb` | 单独运行独立 Benchmark，不依赖完整训练流程 |

最简 notebook 已于 2026-10-21 更新，旧输出已清空，避免把历史结果当作本次实验。较长的“更新后” notebook 保留为历史资料。新实验优先使用最简 notebook，详细说明见 [最新操作指南](guides/haze_density_user_guide.md)。本次未执行正式训练。

## 最简 notebook 的使用方式

1. 将修改推送至远端 `codex-v1` 后，在 Colab 运行第 1—4 节。初始化会明确获取该分支，检查先验版本，并打印提交号。
2. 第 4 节设置 `RUN_NAME`、图像尺寸、batch size 和学习率。配置、manifest、权重、日志和结果保存到 `MyDrive/CrossModal-Defog-OVOD/<RUN_NAME>/`；配置已存在时不会被重写。
3. 首次训练在第 5 节使用 `EPOCHS=5`、`RESUME=False`。继续训练时设 `RESUME=True`，并把 `EPOCHS` 增大到目标总轮数，例如 50。新实验应换 `RUN_NAME`。
4. 第 6 节从 `best.pth` 读取模型、尺寸和先验参数，在验证集展示结果，每次保存到新目录。
5. 第 7 节仅在方案确定后做测试集评估。它显式传递该实验的数据路径和划分，输出到新的空目录。
6. 第 8 节运行独立 Benchmark。第 1、3、4 节准备好后可直接执行；自动复用 RSHaze+ 清晰图和当前实验的合格 `best.pth`，将真值、结果、性能、图表保存在 Drive 的 `benchmarks/<时间戳>/`。权重缺失时学生正确性为 NOT EVALUATED；四个 ROI 仅属一个地理组。

运行时重建后，用原 `RUN_NAME` 和设置完成第 1—4 节，即可续训或直接看图。旧的无版本标记或 v2 权重不能续训为同一 v3 实验。评估指标衡量伪标签拟合，不能独立证明真实物理雾密度准确。

## 本次验证范围

原 7 个代码 cell 使用临时合成图片验证了解压、固定配置与划分、防覆盖、权重加载和验证集可视化；两次预览生成 12 张有效 PNG。训练命令、续训限制和评估传参通过模拟调用检查。Notebook 中的静态检查、40 项单元测试（无跳过）、先验检查和小张量 smoke test 已在本地 CPU 实际通过。新增第 8 节的验证记录见 [阶段 2 审查](stage_2_research_review.md)；尚未在 Colab 内核实际运行。

Colab 挂载、Git 下载和依赖安装使用模拟调用验证分支逻辑，未实际连接 Colab；没有运行真实 RSHaze+ 训练或 GPU 验证。这些检查不能作为正式实验成绩。

## 历史 notebook 和当前代码配合时注意什么？

1. 确认 Colab 加载的是本次修正后的代码，`PRIOR_VERSION` 应为 `spatial_minmax_valid_boundary_v3`。
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
