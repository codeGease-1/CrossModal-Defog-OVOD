# 语义辅助校正 CPU 诊断

结论与边界见 [中文报告](../../docs/语义辅助校正阶段诊断报告.md)。当前所测语义未通过稳定性筛查；没有完整神经训练，没有新的确认测试，没有使用 GPU。

源码入口：`scripts/semantic/`；新模块：`src/models/haze_density/semantic_prior.py`；配置：`configs/semantic_calibration.json`。历史源码、权重及缓存保留。

实际执行（仓库根目录；本机用 `.venv/Scripts/python.exe`）：

```powershell
.venv/Scripts/python.exe scripts/semantic/test_semantic_prior.py
.venv/Scripts/python.exe scripts/semantic/run_diagnostics.py --stage prepare
.venv/Scripts/python.exe scripts/semantic/run_diagnostics.py --stage probe
.venv/Scripts/python.exe scripts/semantic/run_diagnostics.py --stage diagnose
.venv/Scripts/python.exe scripts/semantic/benchmark_cpu.py
.venv/Scripts/python.exe scripts/semantic/real_reference_diagnostics.py
.venv/Scripts/python.exe scripts/semantic/extra_water_probe.py
.venv/Scripts/python.exe scripts/semantic/render_evidence.py
```

`prepare` 不覆盖现有协议。完整重现应使用另一个仓库副本及新输出目录，先复制固定语义权重目录与旧真实参考图，不能删除本次证据。额外水体脚本固定指向本次目录，仅作开发补充。所有训练样本的场景/雾版本由 `scene_manifest.json` 和配置确定。

`protocol_initial_probe.json` 保存早期探针版本；`protocol_before_other_fallback.json` 和 `initial_other_class_trial/` 保存 other 条件类开发试验；最终源码以 `final_source_manifest.json` 与源码 ZIP 为准。首次 probe 在保存数值后因绘图 OpenMP 冲突退出，首次 diagnose 在缓存完成后因 MKL/torch 冲突退出；原日志保留，后续以缓存恢复。修复后 probe 的数值原样保留，由 PIL 单独渲染。不要把早期图形阶段失败误认为语义模型未执行。

`conditional_calibration.json` 是训练集拟合的冻结标定，`fixed_weights.json` 是同一训练集精确优化的固定三权重。`frozen_candidate_manifest.json` 记录供 C/D 共用的候选缓存，不含目标/清晰参考；`diagnostic_cache/` 包含离线目标，仅供诊断，不是部署输入。

神经网络没有训练权重或续训状态；只有契约测试的一次 CPU fixture 更新。`inference_contract.json` 中的随机网络不用于效果比较。恢复状态见 `completion.json`。模型权重、图、ZIP 和缓存留磁盘，不纳入 Git。
