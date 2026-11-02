# 三先验响应的离线可行性检查

2026-11-02。仅 CPU，无训练、无新确认测试。结论和限制见 [可行性复核](../../docs/research/三先验校正与物理约束可行性复核_20261102.md)。

实际运行命令（仓库根目录，Windows）：

```powershell
.venv/Scripts/python.exe scripts/semantic/audit_physical_feasibility.py
```

复现须使用新输出目录，避免覆盖本次计划与证据：

```powershell
.venv/Scripts/python.exe scripts/semantic/audit_physical_feasibility.py --output experiments/physical_feasibility_NEW
```

- `plan.json`：计算前保存的开发来源、扰动、阈值、源码哈希；无按结果改门槛。
- `manifest.json`：48 个源样本的原景组与清晰/有雾/目标数组哈希。
- `per_case.csv`：432 行原始响应、条件反演误差、有效覆盖和边界残差。
- `results.json`：汇总、CPU 环境/耗时、单图不唯一反例；只有离线真值基线，没有训练好的模型。
- `artifact_sha256.json`：运行结束时的产物哈希；本 README 是结果解释文件，后添加，不在该运行清单内。

清晰图、A、t 仅用于离线诊断。条件反演指标忽略分母绝对值 <0.01 的位置，覆盖率必须同时阅读；它不是全图预测成绩。已保存多个条件阈值覆盖率，0.01 不是确认过的物理可靠性门槛。不同先验/处理条件的有效像素可能不同。

程序化场景名称未作为语义。这里证明的是某些关系成立、某些反演敏感，并未证明自然遥感雾强度估计有效。
