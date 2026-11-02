# V100 受限候选校正实验

用户新增授权：使用已连接 V100 实际训练，以相近或更少参数接近同骨干 RGB 回归。旧一小时历史额度不限制本轮新授权；只使用既有实例，不修改云资源、密钥或旧结果。

预先选定容差：像素 MAE 相对 RGB32 增加不超过 5%，局部相关下降不超过 0.01，清晰响应增加不超过 0.005。报告场景成组置信区间，不按确认测试挑模型。

本轮已完成。新24原景 nominal 的三种子均值：校正 MAE .013415，RGB32 .014010，相对差 −4.245%，95% 场景区间 [−8.705%, +.029%]；三项容差均通过。此为三个模型成绩的均值，不是集成预测，种子43单独相对其RGB对照退步18.44%。同输入同容量回归 .012471 更好；未知A、显示域和旧S端点均失败。完整256²推理9.25ms，RGB32 4.48ms；少参数不等于更快。物理项关闭，无独立物理贡献结论。

正式模型：`corrector16`（218,358 神经参数）、`direct_same16`（219,137，同样 11 通道 RGB 派生输入）、`rgb32`（867,377，纯 RGB）。校正和固定融合分开拟合；无自由 H 校正输出，无物理、语义、去雾、检测分支。

训练用历史 48 个程序化训练原景、16 个开发原景，256²，各 32 个版本；独立 H=1−t。正式轮每模型三个种子、18 轮，相同 AMP、batch8、Adam 与数据配对。FP32 单种子开发试验单独保留，不混算成绩。

云端隔离根目录：`/mnt/CrossModal-Defog-OVOD/experiments/learned_correction_20261102`。

对应中文报告：[V100受限候选校正与RGB对照报告](../../docs/V100受限候选校正与RGB对照报告_20261102.md)。正式训练使用 float32 合成显示 RGB，不先量化为 PNG；文件推理示例为 8 bit RGB。H 始终表示协议中的有效遮蔽，不能解释为实测大气雾浓度。

- `source` / `run01`：原 FP32 开发试验不可变版本。
- `source_amp`：未启动训练的中间工程快照。
- `source_formal` / `formal_amp`：正式统一 AMP 版本与结果。
- `pipeline.py`：顺序完成训练/开发评价，停在方案冻结和确认测试之前。
- `cloud_prechange.json`：旧 Git 状态、HEAD、18 份旧 latest 哈希。
- `source*_upload.zip`：各源码/配置版本的精确字节备份。
- `tools_v2`：耗时分解、冻结推理干预与已暴露 S 端点诊断，和训练源码隔离。

正式结果归档后，`final_evidence/formal_amp` 为正式指标/协议/冻结记录，`final_evidence/cost_assets` 为完整耗时及开发集干预，`final_evidence/exposed_original_S` 为已暴露的 S 辅助端点。`analysis_confirmation` 保存场景 bootstrap 结果；`portable_weights` 保存小型推理权重；完整优化器恢复权重保留云端。

`all_seed_weights` 保存全部9份推理权重及SHA256，无确认集选种子。`portable_weights` 中便捷导出按开发候选损失选出校正43、小回归44、RGB43，仅用于便捷执行，不保证单份权重均满足5%容差。`budget_ledger.json` 去除嵌套记账后本轮GPU阶段墙钟约65.58分钟，旧3258.19秒另列；非云账单。`postflight.json` 记录18份旧latest和9份新best不变、GPU作业结束，实例保持原状。

从仓库根目录复现到**新输出目录**：

```bash
python scripts/learned_correction/test_contracts.py
python scripts/learned_correction/run.py --config configs/learned_correction_amp.json --calibration experiments/v2_20261101/calibration.json --output experiments/new_learned_trial --stage train
python scripts/learned_correction/run.py --config configs/learned_correction_amp.json --calibration experiments/v2_20261101/calibration.json --output experiments/new_learned_trial --stage development
python scripts/learned_correction/run.py --config configs/learned_correction_amp.json --calibration experiments/v2_20261101/calibration.json --output experiments/new_learned_trial --stage freeze
python scripts/learned_correction/run.py --config configs/learned_correction_amp.json --calibration experiments/v2_20261101/calibration.json --output experiments/new_learned_trial --stage confirmation
```

导出结果的独立 CPU 文件推理：

```bash
python scripts/learned_correction/predict.py --image experiments/learned_correction_20261102/final_evidence/cost_assets/input_rgb.png --weights experiments/learned_correction_20261102/portable_weights/corrector16_inference.pth --calibration experiments/learned_correction_20261102/final_evidence/cost_assets/calibration.json --output experiments/new_rgb_inference --device cpu
```

同一 PNG 可用于 `audit_export.py` 的已训练权重依赖隔离检查；`audit_results.py` 核验冻结源码 ZIP、原景分组、输入/目标哈希和固定融合。`analyze.py` 只读取离线指标，`plot_results.py` 生成标准静态科研图，不回写模型。

需 CUDA 环境。原运行的完整恢复入口会核对源码、配置和标定哈希，已完成模型跳过。best/latest 含优化器、CPU/CUDA/打乱 RNG 及 AMP scaler；只有在同一源码快照运行时才能恢复。保存权重 JSON/元数据与图片均不逐图拉伸。

正式报告将区分同协议受控结果、域外压力结果和既有原始 S 次要终点。程序化场景不是自然遥感真值；既有真实地理分组和监督资格限制继续保留。
