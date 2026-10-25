# 精选实验归档

本目录按明确的文件清单纳入 Git，当前归档 `spatial_v3_run01` 的 5 轮对照和累计 50 轮续训结果，约 66 MB。新增实验仍默认忽略，需要逐次评估后加入清单。

入口：[50 轮结果与学习曲线](spatial_v3_run01/运行结果.md)、[5 轮对照](spatial_v3_run01/snapshots/epoch_005/README.md)、[文件与 SHA256 清单](artifacts_manifest.json)。

## 保留的模型

| 文件 | 轮次 | 保留理由 |
|---|---:|---|
| [5 轮 best.pth](spatial_v3_run01/snapshots/epoch_005/best.pth) | 5 | 独立雾强度排序对照；包含模型、Adam 和 AMP 状态 |
| [当前 best.pth](spatial_v3_run01/checkpoints/best.pth) | 48 | 50 轮训练期间验证集伪标签 MSE 最低的模型 |
| [当前 latest.pth](spatial_v3_run01/checkpoints/latest.pth) | 50 | 保存训练进度，供后续断点续训 |

两次训练都值得保留：验证 MSE 从 0.007214 降至 0.006717，测试集伪标签拟合 MAE 从 0.064042 降至 0.060457；独立 Benchmark 的均匀雾强度排序准确率从 **98.125% 降至 77.500%**，空间区域 Spearman 从 0.9532 略升至 0.9556。更长训练改善了伪标签拟合，独立排序能力下降。这些结果只覆盖一个地理组，场景内容与训练集的重叠未核验。

## 归档范围

| 内容 | 选择理由与范围 |
|---|---|
| 配置、固定划分、`code_*.json` | 保留参数、样本清单、源码版本与文件哈希 |
| 1—50 轮训练 CSV、5 轮原始 CSV、模型信息、训练总结、学习曲线 | 核对续训连续性、收敛过程和模型选择依据 |
| 两次 930 张测试图的指标报告 | 完整保留拟合误差和预测分布，不仅保留表现较好的一次 |
| 两次 128 样本 Benchmark 的 JSON、全部 CSV、性能原始计时、状态及执行日志 | 保留三种方法、全部条件、逐图数值和失败样例；20 样本最小闭环保留输入清单、执行状态与日志 |
| 两次验证可视化，共 12 张 | 保留原 notebook 每子集前 2 张的完整固定样本集 |
| 测试可视化，共 12 张 | 每次测试、每子集按文件名字典序保留前 2 张，固定配对比较，不按误差大小筛选 |
| Benchmark 对比图，共 24 张 | 每次固定保留 4 个 ROI × 3 种方法；`G_test_1053/regions`、`G_test_1072/gradient`、`G_test_1126/blob`、`G_test_1328/blob`，按场景和模式覆盖选择 |
| Benchmark 排序曲线，共 32 张 | 两次报告中的全部 16 张排序曲线均保留，直接展示排序退化 |
| notebook 执行日志、状态、核验记录及续训说明 | 证明执行状态，记录 CUDA 配置修复和续训来源 |

保留三份原始 checkpoint，直接兼容现有加载脚本。5 轮快照的 `best.pth` 与 `latest.pth` 保存相同的模型、优化器和 AMP 状态，因此 Git 只保留其中的 `best.pth`。

继续忽略合成样本 `.npz`、逐像素预测 `.npy`、数据集原图、重复 notebook 输出副本、重复 checkpoint、最小闭环重复报告和未选中的批量图片。这些文件仍保存在云实例上；本次仅调整 Git 纳入范围。完整图版报告的本地备份名为 `report.full.local.md`。

## 溯源与复核

`artifacts_manifest.json` 记录本次选择的理由、模型对应轮次、固定样例清单，以及每个归档文件的大小和 SHA256；清单自身不计算自身哈希。`snapshots/epoch_005/snapshot.json` 是完整本地快照的历史记录，包含部分未纳入 Git 的副本；本次 Git 文件范围以归档清单为准。

[.gitattributes](../.gitattributes) 禁用实验归档的自动换行转换，保留原始 CSV 换行及日志空白，使这些哈希在不同系统检出后仍可核验。

日志、checkpoint、配置和原始结果 JSON 中的绝对路径按运行时原样保留。5 轮 Benchmark 记录的 `checkpoints/best.pth` 当时指向 5 轮模型，其 SHA256 与现在归档的 `snapshots/epoch_005/best.pth` 一致。读取历史结果时应按哈希辨认权重。

Git 中不包含 JSON 引用的逐像素预测数组与原始图像。需要重新生成图像或计算新指标时，按 [Benchmark 指南](../docs/guides/haze_density_benchmark.md) 使用已记录的源图 SHA256、ROI、合成配置、随机种子及对应模型，在新的空输出目录重建；现有 CSV/JSON 保留了已完成实验的全部原始指标。

在项目根目录可核验归档文件：

```bash
python - <<'PY'
import hashlib, json
from pathlib import Path
root = Path('.')
manifest = json.loads((root / 'experiments/artifacts_manifest.json').read_text())
for item in manifest['files']:
    path = root / item['path']
    assert path.stat().st_size == item['size_bytes'], path
    assert hashlib.sha256(path.read_bytes()).hexdigest() == item['sha256'], path
print('All archived files verified.')
PY
```
