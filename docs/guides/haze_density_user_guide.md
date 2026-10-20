# 雾密度模块操作指南

更新于 2026-10-20。所有终端命令从项目根目录执行；在 Colab 中可用 `!python ...`，环境准备见 [Colab 指南](colab.md)。先阅读 [入门原理](haze_density_explained.md)，理解伪标签的含义。

## 1. 环境与检查

本地阅读和纯 Python 划分检查不需要 PyTorch；模型运行需要：

```bash
python -m pip install -r requirements.txt
python scripts/utils/static_check.py
python -m unittest discover -s scripts/test -p test_review_regressions.py -v
python scripts/test/test_physical_prior.py
python scripts/utils/smoke_test.py
```

回归检查会明确跳过缺少 PyTorch 的项；跳过不是通过。不要直接对全部历史 `test_*.py` 使用 pytest，它们大多是需要真实数据/权重、手动运行的验证脚本。

`smoke_test.py` 现在执行真实的小尺寸前向、反向传播和优化步骤；它使用较小的基础通道数，不能替代正式配置训练。需要完整模型的更多历史检查时，可另行运行 `scripts/test/test_model.py`，注意其中包含较大的 CPU 张量。

## 2. 数据准备

数据来源：[RSHaze+ 官方发布页](https://zenodo.org/records/13837162)。解压后应包含：

```text
datasets/RSHaze+/
  RSHaze_G/train/{synhazypng,cleanpng}/
  RSHaze_G/test/{synhazypng,cleanpng}/
  RSHaze_L/train/{synhazypng,cleanpng}/
  RSHaze_L/test/{synhazypng,cleanpng}/
  RSHaze_S/train/{synhazypng,cleanpng}/
  RSHaze_S/test/{synhazypng,cleanpng}/
```

含雾与清晰图按同名文件配对。当前雾密度损失只使用 `synhazypng`；清晰图、透射率、大气光、NIR 不参与该损失。

使用默认目录时生成并核验固定划分：

```bash
python scripts/data/generate_rshazeplus_split.py
python scripts/data/verify_rshazeplus_split.py
python scripts/test/test_rshazeplus_dataset.py
```

这些历史核验脚本按默认路径和历史规模编写。若数据存放位置不同，先修改配置；需要自定义位置生成 manifest 时可以在 Python 中调用：

```python
from scripts.data.generate_rshazeplus_split import generate_split

generate_split(
    root='/your/path/RSHaze+', val_ratio=0.1, seed=42,
    output_file='experiments/haze_density/rshazeplus_split.json',
)
```

保存 manifest 后不要在同一实验中重新随机划分。缺少 manifest 时，Dataset 会用固定种子生成互补 train/val，并在指定路径保存；已有 manifest 与数据不符或 train/val 重叠时应排查原因，不能用删文件来掩盖问题。

历史规模是 6174/686/930，以实际扫描为准。训练随机裁剪，原图需不小于 `image_size`；验证/测试按双线性插值缩放。训练 DataLoader 使用 `drop_last=True`，会丢弃当轮最后不足一批的样本。

## 3. 运行 5 轮短训练

先看 `configs/haze_density.yaml`。这是本次整理后实际可读取的平铺配置；旧版嵌套 YAML 不再适用。

```bash
python scripts/train/train_haze_density.py --config configs/haze_density.yaml --epochs 5
```

这份配置默认输出到：

```text
experiments/haze_density/checkpoints/spatial_v2/best.pth
experiments/haze_density/checkpoints/spatial_v2/latest.pth
experiments/haze_density/results/spatial_v2/train_log.csv
experiments/haze_density/results/spatial_v2/training_summary.txt
experiments/haze_density/results/spatial_v2/model_info.txt
```

`best.pth` 按验证集 MSE 最低选择；`latest.pth` 用于断点恢复。日志每个 epoch 结束后追加保存。训练脚本目前使用固定学习率 Adam 与 MSE，没有实现学习率调度器或 TensorBoard 写入。

通过 `--config` 才读取 YAML；不传时使用脚本 `DEFAULT_CONFIG`，其输出路径仍为 `formal/`。命令行显式选项覆盖 YAML，例如：

```bash
python scripts/train/train_haze_density.py --config configs/haze_density.yaml --epochs 5 --batch_size 2 --num_workers 0
```

## 4. 正式训练与恢复

若短训练正常，可以继续到总共 50 轮：

```bash
python scripts/train/train_haze_density.py --config configs/haze_density.yaml --epochs 50 --resume experiments/haze_density/checkpoints/spatial_v2/latest.pth
```

`--epochs 50` 表示目标总轮数，必须大于已保存轮数。恢复时会检查关键数据和先验配置，恢复优化器、最佳验证值和 AMP scaler；它不是逐位复现随机过程的恢复，随机数/DataLoader 状态尚未保存。

若要重新开始一次独立实验，使用新的两个输出目录：

```bash
python scripts/train/train_haze_density.py --config configs/haze_density.yaml --epochs 50 --checkpoint_dir experiments/haze_density/checkpoints/run02 --result_dir experiments/haze_density/results/run02
```

已有 checkpoint 或已有新训练日志的目录不会被新训练覆盖。请保留每次使用的代码版本、配置、manifest 和随机种子。

## 5. 看哪些现象？

- `train_loss` 和 `val_loss` 是相对伪标签的 MSE；关注趋势和两者差距。
- 预测/标签应有限且在 0～1 之间；不要求每批都恰好出现 0 和 1。
- 看含雾原图、先验图、预测图、误差图，尤其检查白屋顶、平坦地面、水体与图像边缘。
- 灰度显示应固定范围 0～1。逐图自动拉伸对比度会让很小的差异显得很明显。
- 旧的 `ReLU → Sigmoid` 输出下限 0.5 问题已在现有 Decoder 中修过，不要重复运行历史修复脚本。

## 6. 方案确定后评估测试集

```bash
python scripts/evaluate/evaluate_haze_density.py --checkpoint experiments/haze_density/checkpoints/spatial_v2/best.pth --dataset_root datasets/RSHaze+ --output_dir experiments/haze_density/results/spatial_v2_test
```

输出包含 `test_metrics.txt` 和各子集拼图，顺序是含雾图、先验、预测、误差。误差图会按每幅图的最大误差放大显示；亮度不能用于跨图比较绝对误差，应看数值指标。

每次评估用新的空目录。默认每子集最多 16 张，是遍历时最先遇到的样本，不是随机代表性抽样。评估目前会在 CPU 内存中汇总所有预测/标签，内存不足可先使用默认 256 尺寸；流式评估尚待实现。

报告中的 MAE、RMSE、Pearson 只衡量对当前伪标签的拟合。测试集不要反复用于挑选超参数。先验参数来自 checkpoint 配置；报告记录评分版本与分辨率。

可选：验证集预测分布审计，从 checkpoint 配置读取数据路径和先验参数：

```bash
python scripts/evaluate/audit_prediction_distribution.py --checkpoint experiments/haze_density/checkpoints/spatial_v2/best.pth
```

该历史审计脚本仍写入固定审计目录，运行前自行备份旧审计产物；不能把多次输出当作独立保留的实验。

## 7. 单张图推理示例

在已安装依赖的 Python 或 Colab 中执行；将 `hazy.png` 改为自己的图片路径：

```python
import torch
from PIL import Image
from torchvision.transforms import functional as TF
from src.models.haze_density import HazeDensityNet

checkpoint_path = 'experiments/haze_density/checkpoints/spatial_v2/best.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoint = torch.load(checkpoint_path, map_location='cpu')
config = checkpoint.get('config', {})
model = HazeDensityNet(
    base_channels=config.get('base_channels', 32),
    use_sigmoid=config.get('use_sigmoid', True),
).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

size = config.get('image_size', 256)
image = Image.open('hazy.png').convert('RGB').resize((size, size), Image.Resampling.BILINEAR)
x = TF.to_tensor(image).unsqueeze(0).to(device)
with torch.no_grad():
    density = model(x)[0, 0].cpu()
Image.fromarray((density.numpy() * 255).clip(0, 255).astype('uint8')).save('density.png')
```

输入保持 0～1，不使用 ImageNet 均值方差标准化。此示例统一缩放，不能声称保持了任意原图分辨率的细节。单张评分均值只能作探索性统计，尚不能跨场景直接分为轻雾、中雾、重雾。

## 8. 旧模型怎么办？

先验归一化已修正，新 checkpoint 标记为 `spatial_minmax_v2`。没有这一标记的旧模型不会被当作同一实验继续训练；可以保留作历史对照。评估旧权重会提示版本差异，得到的是相对新标签的探索性结果，不能复现旧表格。

本机现有权重实际在 `experiments/haze_density/checkpoints/best.pth`，来源未核实。本次没有重命名、覆盖或训练它。两个已有 notebook 也未改动，见 [Notebook 使用说明](../notebook_update_guide.md)。
