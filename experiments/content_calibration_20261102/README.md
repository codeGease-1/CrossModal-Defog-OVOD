# 内容条件候选校正：独立 CPU 可行性版本

报告：[中文报告](../../docs/三先验内容校正独立实验报告_20261102.md)。实现可运行；本次共同误差门槛失败，停止扩大训练。无新 GPU 训练或确认测试。

从仓库根目录运行（Windows 的 Python 路径）：

```powershell
.venv/Scripts/python.exe scripts/content_calibration/test_contracts.py
.venv/Scripts/python.exe scripts/content_calibration/run_cpu.py --output experiments/content_calibration_20261102/new_cpu_run
.venv/Scripts/python.exe scripts/content_calibration/diagnose_controls.py --run experiments/content_calibration_20261102/new_cpu_run --output experiments/content_calibration_20261102/new_diagnostics
.venv/Scripts/python.exe scripts/content_calibration/predict.py --image experiments/content_calibration_20261102/cpu_run/input_rgb.png --run experiments/content_calibration_20261102/cpu_run --output experiments/content_calibration_20261102/new_prediction
```

Linux 可将 `.venv/Scripts/python.exe` 换成正确环境的 `python`。本轮未向云端同步新代码，不能假定云端已有入口。

所有输出目录均保护已有结果。主循环可用 `--resume` 恢复已完成阶段；源码/配置不一致时拒绝继续。保存的 `cpu_run` 已完成并通过一次实际 resume 检查，无需重跑。

- `handover_snapshot.json` / `preexisting_changes.zip`：接手时 87 个未提交文件清单、SHA256 及备份。
- `cloud_audit.json` / `cloud_comparison.json`：实际 SSH 核验、原始及换行归一后比较、258 项保护清单，无同步。
- `historical_evidence.json`：两轮 v2 原指标及累计 GPU 账本。
- `cpu_run/protocol.json` / `train_manifest.json` / `development_manifest.json`：数据、种子、源码版本和输入哈希。
- `cpu_run/calibration_content*.json` / `fixed_weights.json`：本轮非参数权重。没有神经训练 checkpoint。
- `cpu_run/development_metrics.json` / `candidate_diagnostics.json` / `content_proxy_errors.csv`：逐候选、共同偏差、包络、离线下界、排序和内容代理误差。
- `diagnostics/results.json`：物理软项的重加权等价性、域外失败、内容打乱、完整固定推理成本和停止决定。
- `standalone_inference`：无真值 API 的实际 RGB 文件推理。
- `delivery_manifest.json` / `source_snapshot.zip`：最终可复核交付和源码快照；大文件留本地，不自动纳入 Git。

所有 H 图固定 [0,1] 色标。清晰图、t、A 只在训练/离线诊断使用；不进入推理候选缓存。旧实验和 latest 权重均保留。
