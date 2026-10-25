# Notebook 使用与迁移说明

更新于 2026-10-25，当前分支为 `v1-nocolab`。

| 文件 | 定位 |
|---|---|
| [最简流程 Markdown](雾密度网络_最简实验流程.md) | 恒源云等平台的终端步骤：环境、持久化数据与配置、训练、续训、评估 |
| [最简流程 notebook](雾密度网络_最简实验流程.ipynb) | 同一流程的 Jupyter 入口，含验证集可视化和第 8 节 Benchmark |
| [独立 Benchmark notebook](haze_density_benchmark.ipynb) | 内嵌代码，独立评测，无需训练；必须设置持久化输出路径 |
| `雾密度网络_更新后.ipynb` | 旧实验的历史记录，保留原命令和输出，仅作溯源 |

## 当前 notebook 的用法

1. 在服务器获取本分支源码，选择已安装项目依赖的 Python 内核。第 1 节检查现有环境与先验版本，默认要求 CUDA；不会自动更新源码。
2. 第 3 节指定 `DATASET_ROOT`，指向直接包含 `RSHaze_G/L/S` 的目录。数据应事先上传和解压。
3. 第 4 节指定持久化的 `OUTPUT_ROOT`、单个目录名 `RUN_NAME` 和训练参数，调用与终端共用的 `prepare_experiment.py`。已有配置/manifest 会核对并复用。
4. 第 5 节首次用 `EPOCHS=5, RESUME=False`；继续时设 `RESUME=True` 和更大的目标总轮数，例如 50。新训练需新 `RUN_NAME`。
5. 第 6 节读取 `best.pth`，在验证集显示固定 0～1 色标的预测；每次输出到新的 `validation_previews/` 子目录。
6. 方案确定后再运行第 7 节测试与第 8 节独立 Benchmark。Benchmark 复用本次数据、有效 v3 权重，结果写入实验的 `benchmarks/`；缺少官方小样本时，联网下载需要手动打开开关。

重启后重新设置原路径与参数，执行第 1—4 节，然后续训或直接查看已有权重。若仅作本地 CPU 检查，第 1 节可设 `REQUIRE_CUDA=False`。当前流程无需云盘挂载或固定平台路径。

## 维护内嵌 Benchmark

修改源码后依次运行：

```bash
python scripts/benchmark/build_benchmark_notebook.py
python scripts/benchmark/sync_minimal_notebook.py
python -m unittest scripts.test.test_minimal_notebook_benchmark -v
```

第一个脚本重建独立 notebook 和代码包哈希；第二个只更新最简 notebook 中带标签的第 8 节，不修改训练设置。内嵌包只含公开代码与配置，无数据、权重或凭据。发布源码前重建，避免 notebook 携带过期代码。

## 历史边界

旧 notebook 仍保留历史环境、源码修复和数据删除命令，不能作为当前分支的可执行指南。不要重复运行旧 `fix_decoder_relu.py`、批量路径修复或源码回退单元。历史验收记录中的平台名称和文件哈希保留当时原貌。

本次迁移的 CPU 检查不代表已在算力平台训练。标签仍是 `spatial_minmax_valid_boundary_v3` 的相对评分伪标签，算法和实验结论未因平台迁移改变。

## 本次迁移验证（2026-10-25）

- 本地 CPU 回归共 79 项通过，无跳过；最终 notebook 更新后，8 项平台/集成测试再次通过。
- 最终内嵌包的 76 个文件逐一核对仓库源码 SHA256，通过；解包到仓库之外执行 71 项原回归，全部通过。
- 临时生成三个子集的 PNG，实际通过终端 CLI 准备配置、重复准备而不改写配置/manifest、训练 1 轮、恢复至第 2 轮、测试评估并生成三个子集的图片。它们是流程测试，不能作为模型效果证据。
- 静态检查、两份当前 notebook 的代码语法和空输出检查、小张量前向/反向检查通过。
- 本机是 CPU 版 PyTorch；`--require-cuda` 正确以失败退出。完整依赖检查也正确报告本机缺少 `cv2`。本机 Windows 的 notebook Matplotlib 预览尝试遇到重复 OpenMP 运行库而中止，未通过动态验证；测试集的 PIL 图片输出已实际生成。未运行真实 RSHaze+ 或云端 GPU 训练。

本地流程日志保存在 Git 忽略的 `experiments/cloud_workflow_check_20261025_032656_377899/`。历史 `docs/validation/v1_20261022/` 的原始验收证据保持原样。
