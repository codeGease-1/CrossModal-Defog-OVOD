# v2 塌缩原因与学生重构的继续实验

这是第一轮 v2 后的独立目录，不覆盖原结果。三个变体的三种子训练、新确认集、压力测试和空间消融均已完成。结论：动态区域加权在受控协议中有效，但RGB必要对照整体更强，原始S局部退步仍在，暂不切换主流程。见 [中文结果报告](../../docs/v2深入探索与学生重构验证报告.md)。

## 本轮问题

1. 小门控是否根本无法反向传播、无法学会选择？
2. softmax 饱和是否足以解释上一轮结果？
3. 复用原学生的更大上下文结构后，能否学会实际有益的空间权重？
4. 若有改善，三先验约束相对同骨干 RGB 直接回归是否值得保留？

## 模型关系

第一轮 v2 新增小门控，没有把原 `HazeDensityNet` 重构成权重估计器。本轮 `StudentWeightGate` 复用原 Encoder、三路 MultiScale、Fusion 和解码中间特征结构，替换单通道密度头为三权重头，保留 RGB/三候选/原始强度到输出头的直接路径。原模型文件、训练脚本和旧权重不改。

重构只复用结构，不加载旧伪标签权重，以避免旧随机数据划分带来的外部泄漏。输出权重经过 softmax，最终图是三候选的凸组合，无自由残差。独立目标为已知标量 t 对应的 H=1-t，监督的是最终图，不把旧固定融合当答案。

详细数据流见 [模型关系说明](../../docs/v2模型关系与重构说明.md)。

## 冻结协议与对照

- 完全复用 v1.5 的48训练/16开发原始场景、第一轮 v2 的候选图标定及精确固定权重，后续没有重新拟合标定。
- 前轮固定融合与小门控作为冻结参考。本轮新增的三个训练变体是温度门控、重构学生权重网络、同骨干 RGB 直接回归；各3种子、18轮，优化器、学习率、batch和独立目标相同。
- 温度门控保持原小门控结构，温度从1改为2，调整初始 bias 使起始融合仍为0.5/0.3/0.2。它用于隔离温度因素，不同时添加熵正则或损失重加权。
- 重构学生与温度门控温度相同，但容量/上下文不同。不能进一步拆分其每一个多尺度模块的独立贡献。
- RGB 对照仅访问RGB，去除候选图和原始先验输入、直接回归H；它的输出限制和初始化与凸融合不同，需在报告中承认。仅检验三先验价值，不自动进入主流程。
- 新确认16原始场景，种子偏移80000，全部雾版本同组。主终点在线性化 sRGB 合成。额外预声明未见大气光和显示RGB域作为压力测试，不能在这些结果上调参。
- 原始 S 次要诊断另选8个未用成员，排除旧清单与解码重复；先核验原尺寸逐通道重建及方向，再用 `H_eff=1-mean_RGB(t_c)` 评价。真实来源仍只有一个未核验地理组，不给地理泛化置信区间。
- 新真实S和RGB必要对照的补充计划在任何新测试结果出现前登记，见 `original_S/plan.json` 与 `rgb_control/plan.json`。原始主协议不重写。
- 空间控制在测试前登记，见 `spatial_controls/plan.json`：冻结模型推理后，把每图空间权重换成该图均值，或在图内打乱权重向量，检验收益是否真来自区域对应关系；另算仅供离线诊断的每图最优常数权重。无额外训练，不用它挑模型。

## 复现

V100/PyTorch环境、`/mnt/RSHaze+.zip` 和前轮 `experiments/v2_20261101` 依赖齐全后，在仓库根目录执行：

```bash
bash scripts/v2/reproduce_investigation.sh
python scripts/v2/summarize_rgb_comparison.py
python scripts/v2/plot_investigation_training.py
python scripts/v2/report_investigation_tables.py
python scripts/v2/finalize_investigation.py
```

配置路径是 `configs/v2_investigation.json`。辅助脚本固定使用其中的输出目录。完整从零复现应在另一个仓库副本里运行，保留本次结果；不要为了复现而删除已有文件。每轮记录优化器、训练/开发损失、RNG和权重探针，最后以新确认集全部样本权重统计为准。

本次实际发现 GPU 恢复时 `torch.load(map_location='cuda')` 会把 RNG ByteTensor 也移至 GPU，导致 `set_rng_state_all` 报错。为保留冻结研究源文件，单独提供 `resume_compat.py`：加载后仅把 RNG 状态转回 CPU，不改模型或优化器。训练/恢复均推荐用复现脚本里的兼容入口；已有完成标记则跳过。`verify_student_resume.py` 实际在 CUDA 上中断、恢复，验证参数与连续训练逐项相同，见 `resume_fixture.json`。前轮 CPU 恢复核验没有覆盖该 GPU 问题。

历史日志中的 `probe_mean_weights` 只抽取 `vx[::128]`，即4个开发清晰样本，不能把它当全部雾条件的权重分布；`analysis.json` 才是冻结模型的完整确认分布。

单裁块和人工可学习选择题只是优化诊断，不是泛化结果；没有用这些单块参数初始化正式训练。正式三种子均从头初始化。

冻结后独立推理（选择新输出目录，保留原图和历史结果）：

```bash
python scripts/v2/predict_weight_gate.py --image /path/to/image.png \
  --experiment experiments/v2_investigation_20261101 --mode student \
  --output experiments/my_new_weight_prediction --device cuda
```

默认256×256匹配标定分辨率，输出原始浮点 H/三权重/三候选、统一色标图及哈希记录。改分辨率会改变先验的像素尺度，属于未验证用途。推理不会访问透射率标签，所得H仍是受训练成像协议限制的有效遮蔽估计。

## 预算

沿用前轮总预算要求，前轮 v2 训练加标定约363秒。本轮另设3000秒的共同上限，三变体共用，不为RGB对照另外分配预算。此轮账本保守计入预处理、诊断、训练、评价的GPU阶段墙钟时间；与前轮计时口径不同，报告应分开说明。预留300秒评价时间，预算不足时保存进度并停止，不扩大搜索。

## 文件索引

| 文件/目录 | 内容 |
|---|---|
| `protocol.json`, `confirmation_manifest.json` | 新测试前的配置、源哈希、旧标定哈希、场景种子 |
| `preflight.json`, `cost_forecast.json` | 实测参数量/步耗时与训练预测 |
| `diagnostics.json`, `regional_bounds.csv` | 人工选择题、单训练块拟合、地物代理能力边界 |
| `temperature_seed*/`, `student_seed*/` | best/latest、优化器、RNG、逐轮历史 |
| `frozen.json` | 两种权重模型完成后按原开发集冻结；RGB对照另有独立冻结文件，所有新测试都在训练和选模结束后开启 |
| `nominal/`, `unseen_airlight/`, `display_rgb/` | 相同新场景的主评价和压力测试、统一色标图 |
| `original_S/` | 补充计划、成员清单、标签核验、次要终点评价 |
| `rgb_control/` | RGB必要对照的独立计划、权重、冻结和同样本评价 |
| `analysis.json`, `rgb_comparison.json` | 三种子、成组bootstrap、塌缩/地物代理指标 |
| `failure_and_gain_cases.json` | 相对优化固定融合的实际最大改善/退步例 |
| `spatial_controls/` | 完整空间权重、每图均值权重、打乱权重及离线常数权重 oracle 的严格区分 |
| `inference_cost.json` | 包含先验与标定的完整推理耗时 |
| `budget.json`, `completion.json`, `artifact_sha256.json` | 消耗、旧文件未变检查、交付哈希 |

所有强度与权重图使用共同 [0,1] 色标。离线 oracle 不参与推理。清晰样本约束与无雾响应都保留。RGB高亮/低纹理只是困难地物代理，未加入语义、多模态、去雾或检测模块。

`analysis.log` 保留了结果汇总时一次 tuple/list 拼接错误；修复后成功记录在 `analysis_retry.log`。这次错误发生在训练、冻结和评价完成之后，不影响模型参数或逐例原始指标。
