# CrossModal-Defog-OVOD

跨模态语义恢复引导的含雾低质遥感影像开放词汇目标检测。大学生创新训练项目，北京师范大学人工智能学院。

**当前可用的核心是雾密度感知模块及其训练、评估代码。完整的去雾、超分辨率和开放词汇检测链路尚未实现。** 当前输出是物理先验生成的伪标签所定义的相对雾感评分，不是实测雾浓度。

最近更新：2026-10-25。`v1-nocolab` 已切换为通用 GPU 服务器流程，支持恒源云等平台，数据和输出路径可配置。阶段 1 必要修复已完成，详见 [修复报告](docs/stage_1_bugfix_report.md)、[项目状态](docs/project_status.md) 与 [首次审查记录](docs/project_review.md)。

阶段 2 已新增独立雾密度 Benchmark，并实际完成 128 个受控合成样本的 CPU 评价。三先验在本次单地理区域试验中，均匀雾排序准确率为 21.88%，空间区域相关为 0.956；这两种能力不能混为一谈。学生正确性及 GPU 结果尚未评测。**最简实验流程 notebook 第 8 节可直接运行此 Benchmark**。见 [科研审查报告](docs/stage_2_research_review.md)、[复现说明](docs/guides/haze_density_benchmark.md) 与 [独立 Benchmark notebook](docs/haze_density_benchmark.ipynb)。

## 建议阅读顺序

1. [从零理解雾密度分析](docs/guides/haze_density_explained.md)：先理解输入、标签、训练和结果的含义。
2. [最简实验流程](docs/雾密度网络_最简实验流程.md)：从获取 `v1-nocolab` 到训练、续训、评估的终端命令。
3. [GPU 云平台指南](docs/guides/cloud_gpu.md)：环境、持久化磁盘和 Jupyter 配置。
4. [详细操作指南](docs/guides/haze_density_user_guide.md)：数据与训练原理说明。
5. [完整文档索引](docs/INDEX.md)：历史记录与调研资料。

## 目前的处理流程

```mermaid
flowchart LR
    I[含雾 RGB 图像] --> P[暗通道 / 局部对比度 / 颜色差异]
    P --> S[归一化、加权、幂变换、引导滤波]
    S --> T[伪标签 S_final]
    I --> N[HazeDensityNet]
    N --> D[预测雾感评分图]
    T --> L[MSE 损失]
    D --> L
    L -.训练时更新参数.-> N
```

训练时先验规则充当“出题并给参考答案的老师”，网络学习逼近这些参考答案。推理时只需网络和 RGB 图像。物理先验分支不在网络 `forward()` 内。评分在 0～1 之间，数值高表示更符合当前规则定义的雾特征。

## 目录与职责

```text
configs/                    正式训练配置、集成原型配置
src/data/                   图像读取、固定划分、裁剪与翻转
src/models/haze_density/    先验、引导滤波、雾密度网络
src/models/backbone/        SimpleBackbone 集成原型
src/models/density_guidance/ 密度引导注意力
src/models/crossmodal/      密度拼接、特征引导原型
scripts/data/               数据核验、划分、可视化
scripts/train/              8 图过拟合、正式训练
scripts/evaluate/           测试评估、预测分布审计
scripts/test/               模块与回归检查
scripts/utils/              静态检查、环境检查
experiments/                本机/GPU 服务器生成的实验，默认不提交权重
docs/                       指南、状态与历史阶段记录
tools/                      历史迁移辅助工具，日常训练无需运行
```

## 快速开始

在平台的 PyTorch GPU 环境安装依赖，确认 CUDA 可以实际运算：

```bash
python -m pip install -r requirements.txt
python scripts/utils/setup_environment.py --require-cuda
python scripts/utils/smoke_test.py
```

接着按[最简实验流程](docs/雾密度网络_最简实验流程.md)设置数据与持久化实验目录：

```bash
python scripts/utils/prepare_experiment.py --dataset-root /实际数据目录/RSHaze+ \
  --output-root /持久化磁盘/experiments --run-name spatial_v3_run01
python scripts/train/train_haze_density.py \
  --config /持久化磁盘/experiments/spatial_v3_run01/train_config.yaml --epochs 5
```

示例路径需替换。`--epochs 5` 是短训练；正式训练使用保存的配置和 `latest.pth` 续训至目标总轮数。配置、manifest、权重与日志直接写入选定的持久化目录；同名新训练不会覆盖已有权重。Jupyter 可使用[最简 notebook](docs/雾密度网络_最简实验流程.ipynb)。

## 版本与实验结论

先验归一化维度已复核正确；阶段 1 又修复局部均值边界、常量/低精度数值错误、成对增强错位和同名样本误删。新 checkpoint 记录 `prior_version=spatial_minmax_valid_boundary_v3`。公式、固定参数、网络结构和 MSE 目标保持不变。旧 checkpoint 保留为历史产物，旧指标不能当作新流程的成绩。

阶段 1 在本地 CPU PyTorch 环境完成了 40 项单元测试与小张量前向/反向检查，未运行真实 RSHaze+ 训练。阶段 2 的独立 Benchmark 与限制见上方报告；平台迁移不改变这些研究结论。

项目供学术研究使用；代码、数据和未来引用模型的许可需分别核对。
