# CrossModal-Defog-OVOD

跨模态语义恢复引导的含雾低质遥感影像开放词汇目标检测。大学生创新训练项目，北京师范大学人工智能学院。

**当前可用的核心是雾密度感知模块及其训练、评估代码。完整的去雾、超分辨率和开放词汇检测链路尚未实现。** 当前输出是物理先验生成的伪标签所定义的相对雾感评分，不是实测雾浓度。

最近更新：2026-10-21。阶段 1 必要修复已完成，详见 [修复报告](docs/stage_1_bugfix_report.md)、[项目状态](docs/project_status.md) 与 [首次审查记录](docs/project_review.md)。

## 建议阅读顺序

1. [从零理解雾密度分析](docs/guides/haze_density_explained.md)：先理解输入、标签、训练和结果的含义。
2. [操作指南](docs/guides/haze_density_user_guide.md)：准备数据、训练、恢复和评估。
3. [Colab 指南](docs/guides/colab.md)：在云端准备环境并持久保存实验。
4. [完整文档索引](docs/INDEX.md)：历史记录与调研资料。

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
experiments/                本机/Colab 生成的实验，默认不提交权重
docs/                       指南、状态与历史阶段记录
tools/                      历史迁移辅助工具，日常训练无需运行
```

## 快速开始

只看代码和检查语法无需安装 PyTorch。运行网络需要安装 [运行依赖](requirements.txt)，可在 Colab 中进行。

```bash
python -m pip install -r requirements.txt
python scripts/utils/static_check.py
python -m unittest discover -s scripts/test -p test_review_regressions.py -v
python -m unittest discover -s scripts/test -p test_stage1_bugfixes.py -v
```

先按 [操作指南](docs/guides/haze_density_user_guide.md) 将 RSHaze+ 放入 `datasets/RSHaze+/`，在项目根目录执行：

```bash
python scripts/data/generate_rshazeplus_split.py
python scripts/train/train_haze_density.py --config configs/haze_density.yaml --epochs 5
```

`--epochs 5` 是短流程检查。正式训练用新的输出目录或从此次 `latest.pth` 恢复；同一输出目录不会被新训练覆盖。配置现在会真正被训练脚本读取，命令行显式参数优先。

## 版本与实验结论

先验归一化维度已复核正确；阶段 1 又修复局部均值边界、常量/低精度数值错误、成对增强错位和同名样本误删。新 checkpoint 记录 `prior_version=spatial_minmax_valid_boundary_v3`。公式、固定参数、网络结构和 MSE 目标保持不变。旧 checkpoint 保留为历史产物，旧指标不能当作新流程的成绩。

本次在项目内 `.venv` 的真实 CPU PyTorch 环境完成 40 项单元测试，全部通过、无跳过，并通过小张量前向/反向检查。未运行训练实验，本机没有真实 RSHaze+；独立正确性与性能 Benchmark 留给阶段 2，方法改进留给阶段 3。

项目供学术研究使用；代码、数据和未来引用模型的许可需分别核对。
