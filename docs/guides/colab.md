# 在 Colab 运行项目

更新于 2026-10-20。建议先看 [操作指南](haze_density_user_guide.md)。本地负责改代码，Colab 用于模型动态检查和 GPU 实验；Colab 的资源与会话时长以实际运行时为准。

## 1. 获取本次修改后的代码

本次审查只修改本地文件，没有提交或推送远端。请使用包含本次修改的项目副本：可以手动上传到 Colab，或自行提交到仓库后再克隆。单独克隆旧远端不会自动带上本次修正。

远端已更新后，在新 Colab 会话执行：

```python
!git clone https://github.com/codeGease-1/CrossModal-Defog-OVOD.git /content/CrossModal-Defog-OVOD
%cd /content/CrossModal-Defog-OVOD
%pip install -r requirements.txt
```

已存在项目目录时直接进入该目录，并确认代码版本，不要反复 clone 或覆盖自己的实验。检查版本标记与环境：

```python
import torch
from src.models.haze_density.physical_prior import PRIOR_VERSION
print('torch:', torch.__version__)
print('CUDA:', torch.cuda.is_available())
print('prior:', PRIOR_VERSION)  # spatial_minmax_v2
```

## 2. 挂载 Drive，准备数据

```python
from google.colab import drive
drive.mount('/content/drive')
```

把 RSHaze+ 解压成操作指南中的目录。若压缩包已放在 Drive，可使用：

```python
from pathlib import Path
import shutil

archive = Path('/content/drive/MyDrive/dataset/RSHaze+.zip')
destination = Path('/content/CrossModal-Defog-OVOD/datasets')
if not (destination / 'RSHaze+').exists():
    destination.mkdir(parents=True, exist_ok=True)
    shutil.unpack_archive(archive, destination)
assert (destination / 'RSHaze+' / 'RSHaze_G').exists(), '检查压缩包解压后的层级'
```

压缩包可能多包一层目录，按实际结构确认路径；不要因为断言失败就删除整份数据。

## 3. 把输出直接写到 Drive

`/content` 是临时运行时。自动保存到这里的 checkpoint 不能保证在运行时销毁后还在；把输出指向 Drive 才能在下次会话继续使用。

```python
from pathlib import Path
import yaml

run_dir = Path('/content/drive/MyDrive/CrossModal-Defog-OVOD/spatial_v2_run01')
run_dir.mkdir(parents=True, exist_ok=True)
config = yaml.safe_load(Path('configs/haze_density.yaml').read_text())
config['checkpoint_dir'] = str(run_dir / 'checkpoints')
config['result_dir'] = str(run_dir / 'results')
# 使用绝对路径指向持久化的 manifest；数据本身仍可存 /content。
config['split_file'] = str(run_dir / 'rshazeplus_split.json')
config_path = run_dir / 'train_config.yaml'
if not config_path.exists():
    config_path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding='utf-8')
config = yaml.safe_load(config_path.read_text(encoding='utf-8'))
print(config_path)
```

新会话恢复同一实验时使用已保存的配置，不要悄悄改写它。要改关键数据/先验设置，应开新实验目录。

## 4. 固定划分并运行检查

```python
from scripts.data.generate_rshazeplus_split import generate_split
if not Path(config['split_file']).exists():
    generate_split(root=config['dataset_root'], val_ratio=0.1, seed=42,
                   output_file=config['split_file'])
```

```python
!python scripts/utils/static_check.py
!python -m unittest discover -s scripts/test -p test_review_regressions.py -v
!python scripts/test/test_physical_prior.py
!python scripts/utils/smoke_test.py
```

本次本地未执行的 4 项 PyTorch 回归应在此处真实运行，不能仍显示为 skipped。

## 5. 短训练、恢复与正式评估

```python
import subprocess
subprocess.run(['python', 'scripts/train/train_haze_density.py',
                '--config', str(config_path), '--epochs', '5'], check=True)
```

确认日志与可视化正常后，继续到总共 50 轮：

```python
subprocess.run(['python', 'scripts/train/train_haze_density.py',
                '--config', str(config_path), '--epochs', '50',
                '--resume', str(run_dir / 'checkpoints/latest.pth')], check=True)
```

方案确定后做测试评估，每次使用新的空输出目录：

```python
subprocess.run(['python', 'scripts/evaluate/evaluate_haze_density.py',
                '--checkpoint', str(run_dir / 'checkpoints/best.pth'),
                '--dataset_root', config['dataset_root'],
                '--output_dir', str(run_dir / 'test_evaluation')], check=True)
```

显存不足时，减小 `batch_size`；加载进程出问题时，先尝试 `num_workers: 0`。CPU 能做小规模检查，完整实验建议使用可用的 GPU。当前训练脚本没有 TensorBoard 事件写入，不要使用空的 `logs/` 目录来判断训练是否正常。

## 6. 一次实验要留下什么？

保存代码提交号或源代码快照、配置、split manifest、`best/latest.pth`、`train_log.csv`、测试指标与示例图。报告时注明标签为伪标签、先验版本、图像尺寸与是否使用独立验证证据。历史 notebook 的新旧命令差异见 [Notebook 使用说明](../notebook_update_guide.md)。
