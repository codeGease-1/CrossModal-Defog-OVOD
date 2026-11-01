# v1.5 固定标定：证据与复现

结论请先读 [中文报告](../../docs/v15归一化审查与实验报告.md)。这是一个受控退化下有效、原始 S 光谱配对上尚未通过局部非劣要求的原型，不是已完成真实泛化验证的雾密度模型。

本轮三个候选：原版 A、原版加训练集固定单调标定 B、原始线索使用训练集固定分位数范围后再固定单调标定 C。固定融合权重、幂变换、引导滤波保留。没有训练全局分支或动态门控。

## 原始证据

- `protocol.json`：完整冻结配置、源代码哈希、基线提交与环境。实际新增代码用源文件 SHA256 标识；不把未提交代码冒充基线提交内已有代码。
- `scene_manifest.json`：80 个独立程序生成原始场景及哈希。训练 48、开发 16、测试 16；所有版本继承原景划分。
- `external_manifest.json`：24 张真实参考图的压缩包条目和 SHA256，全部真实来源归于同一个保守外部组。
- `audit/archive_audit.json`：G/S/L 每类 8 个参考样本；G/S 共 16 组透射率、大气光和配对重建核验。
- `smoke.json`：原版一致性、均匀仿射加雾下的信息丢失、输出 InstanceNorm 与旧 best 权重的挂钩统计。
- `calibration.json`：部署所需的固定统计量与单调映射表；仅由程序生成训练场景拟合。
- `fit_cache/`、`limits_checkpoint.json`：按原景保存的恢复缓存；这些生成缓存默认不纳入 Git。
- `frozen_selection.json`：在测试前按开发集预定准则冻结的候选和参数哈希。
- `development/`、`test/`、`external/`：逐图指标、信号统计 CSV、汇总、各大气光/亮度分层和失败案例。
- `secondary_diagnostics.json`：跨亮度/大气光的跨图排序、受控原景 bootstrap、256/512 推理成本。其脚本在冻结测试前上传；不改变选模规则。
- `verified_spectral/`：原始 S 配对的补充评价；明确使用 `1-mean_RGB(t_c)`。该端点区别于主实验的标量 t，不参与模型选择。
- `budget.json`：固定标定累计约 28 秒、神经训练 0 秒，不代表云实例总计费时长。
- `pip_freeze.txt`、各阶段 `.log`、`completion.json`、`artifacts_sha256.json`：环境、日志、结论状态和最终文件校验。

空间 Spearman 对 4×4 区域均值计算，只纳入非均匀真值；不对均匀 t 的相关系数伪造分数。整体 MAE 为每图 `abs(mean(pred)-mean(1-t))` 的平均，另有像素 MAE。同场景排序固定大气光和亮度。主 `summary.json` 中跨图排序固定大气光和亮度；中文报告采用 `secondary_diagnostics.json` 的更严格跨条件跨图排序。成对比较的数量不是独立样本数量，真实数据也不是 24 个已核验独立地理区域。t=1 的两种大气光导致输入重复，所有候选使用相同样本权重。

`clear_mean` 在程序生成域表示无新增退化的清晰输入响应，在真实参考图域仅表示未额外加雾的参考输入响应，不能称为真实无雾假阳性率。`bright_response` 依据原始参考 RGB 的高亮掩码，不能当作已标注白屋顶或水面类别性能。

## 云端复现

Python 3.12、PyTorch 2.4.1+cu124、V100 16GB。无需完整解压 7.4 GB 压缩包；本轮实验完整文件约 30 MB。依赖包括 NumPy、Pillow、Matplotlib、PyTorch。第一次 CUDA 运算的额外初始化包含在 `smoke.json` 首样本用时中，不作为正式推理延迟。

从项目根目录执行，使用新的实验目录保留本轮原始结果：

```bash
bash scripts/v15/reproduce.sh experiments/v15_reproduction /mnt/RSHaze+.zip
```

每阶段也可单独执行：

```bash
python scripts/v15/run.py --stage fit --output experiments/v15_reproduction
```

`fit` 复用已完成的原景缓存，累计预算达到 3600 秒会停止并保留缓存；本轮拟合只有 28 秒，不需要申请额外训练。已完成阶段不会重新拟合或重新选模，测试前校验冻结参数 SHA256。源文件或配置改变后会拒绝混用，应另建实验目录。此处“恢复”指固定标定的阶段/原景恢复，没有需要恢复的 Adam、AMP 或神经训练状态。

加载冻结参数进行推理：

```python
import json
import torch
from src.models.haze_density.calibrated_prior import CalibratedPrior

with open('experiments/v15_20261101/calibration.json', encoding='utf-8') as f:
    calibration = json.load(f)
model = CalibratedPrior(calibration, variant='fixed').cuda().eval()
with torch.no_grad():
    strength_map = model(rgb)  # CUDA RGB float tensor B,3,H,W in [0,1]
    overall_strength = strength_map.mean(dim=(-2, -1))
```

该调用用于复现实验，当前参数尚不建议替换原版主流程。旧网络、权重、旧配置和远程原有 notebook 改动均保留。
