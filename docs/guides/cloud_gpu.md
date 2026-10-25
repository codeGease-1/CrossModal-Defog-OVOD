# 在恒源云等 GPU 平台运行项目

更新于 2026-10-25，当前分支 `v1-nocolab`。直接可执行的步骤见[最简实验流程](../雾密度网络_最简实验流程.md)；本页补充环境、路径、远程会话和迁移说明。

## 环境

选择平台提供的 NVIDIA GPU + PyTorch 镜像，使用其中匹配的 `torch` / `torchvision`。安装项目依赖后运行：

```bash
python scripts/utils/setup_environment.py --require-cuda
python scripts/utils/smoke_test.py
```

环境脚本从自身文件位置确定项目根目录，不依赖启动目录或某个平台的挂载点。它检查依赖、GPU 信息和实际 CUDA 张量运算；`--require-cuda` 失败会以非零状态退出。省略该选项可做本地 CPU 检查。

平台预装环境与自己创建的虚拟环境相互独立。始终使用 `python -m pip`，并检查 `which python`。若另建环境，需在该环境安装匹配的 CUDA 版 PyTorch；不要复制 Windows 的 `.venv` 到 Linux。安装命令以 [PyTorch 官方说明](https://pytorch.org/get-started/locally/)和平台驱动为准。

## 数据与持久化

三个目录各自独立配置：

| 路径 | 用途 | 设置方式 |
|---|---|---|
| `PROJECT_ROOT` | 本分支源码 | 终端进入仓库；notebook 自动查找或读取环境变量 |
| `DATASET_ROOT` | 直接包含 `RSHaze_G/L/S` 的数据根目录 | 准备命令的 `--dataset-root`；notebook 同名变量 |
| `OUTPUT_ROOT` | 所有实验的持久化父目录 | 准备命令的 `--output-root`；notebook 同名变量 |

挂载路径和磁盘寿命由平台决定，项目不预设恒源云或其他平台的目录。确认保存规则后，把数据和实验放入持久化磁盘。可通过上传、SFTP 或挂载数据盘准备数据；数据不必在源码目录内。

`prepare_experiment.py` 在 `OUTPUT_ROOT/RUN_NAME/` 生成冻结配置、manifest 和源码哈希记录。训练、续训和评估都使用这份配置。新设置使用新实验名；已有配置不会被覆盖。manifest 检查只核对路径与样本覆盖，不能替代场景/内容重复核验。

## 远程终端与 Jupyter

SSH 或网页终端运行终端版流程。长任务可使用平台已有的 `tmux`：先 `tmux new -s haze`，在会话内启动训练；断开后用 `tmux attach -t haze` 返回。平台停止实例后，终端进程仍会结束，应从持久化的 `latest.pth` 恢复。

Jupyter 使用[当前最简 notebook](../雾密度网络_最简实验流程.ipynb)。内核必须属于安装依赖的服务器 Python；必要时在该环境安装 `ipykernel` 并注册内核。环境变量需在启动 Jupyter 之前设置，或直接编辑 notebook 中的路径。notebook 不自动安装依赖、切换分支或更新源码；更新后重启内核。

## 断点与迁移

`--epochs` 是目标总轮数。恢复同一实验时保持配置、manifest 和绝对路径一致，使用 `checkpoints/latest.pth`；`best.pth` 用于评估。当前恢复检查严格比对数据根目录、split 路径、模型和先验参数。把实验复制到新机器时，优先通过挂载点或符号链接恢复原路径。不要只修改 YAML 后把它当作同一实验继续。

无版本标记或 v2 先验权重不能续训为当前 v3 实验。默认训练脚本在 CUDA 不可用时会选择 CPU，所以云端正式实验应先通过 `--require-cuda`，并检查训练日志中的 `Device: cuda`。

每次实验保留源码/提交号、`code_*.json`、`train_config.yaml`、split manifest、`best/latest.pth`、`train_log.csv`、测试报告与示例图。历史阶段记录和旧 notebook 中的平台命令只作溯源；当前入口以本指南和最简流程为准。
