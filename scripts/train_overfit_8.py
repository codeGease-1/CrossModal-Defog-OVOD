#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
HazeDensityNet 8-Image Overfit Test (Stage 5C)

目标：验证 HazeDensityNet 能否过拟合 8 张真实 RSHaze+ 图像的 Physical Prior。

训练配置:
    - image_size: 256
    - batch_size: 2
    - epochs: 50
    - optimizer: Adam
    - lr: 1e-4
    - loss: MSELoss
    - AMP: True
    - seed: 42

样本选择:
    - RSHaze_G: 2
    - RSHaze_L: 4
    - RSHaze_S: 2
    - 固定随机种子 42

验收标准:
    - 无 NaN/Inf
    - loss 总体下降
    - prediction 与 S_final 越来越接近
    - checkpoint 正常保存
"""

import sys
from pathlib import Path
import time
import random
import csv

# 设置路径
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.cuda.amp import autocast, GradScaler

from src.data import HazeDensityDataset
from src.models.haze_density import HazeDensityNet
from src.models.haze_density.physical_prior import PhysicalPriorModule


# ============================================================================
# 配置
# ============================================================================

CONFIG = {
    'seed': 42,
    'image_size': 256,
    'batch_size': 2,
    'epochs': 50,
    'lr': 1e-4,
    'amp': True,
    'dataset_root': 'datasets/RSHaze+',
    'split_file': 'experiments/haze_density/rshazeplus_split.json',
    'checkpoint_dir': 'experiments/haze_density/checkpoints/overfit_8',
    'result_dir': 'experiments/haze_density/results/overfit_8_samples',
    'samples_per_subset': {'RSHaze_G': 2, 'RSHaze_L': 4, 'RSHaze_S': 2},
}

# 可视化 epoch
VIS_EPOCHS = [1, 5, 10, 20, 30, 40, 50]


def set_seed(seed: int):
    """设置随机种子"""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def print_separator(title: str):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def select_8_samples(dataset: HazeDensityDataset) -> list:
    """
    从数据集选择 8 张样本

    Returns:
        indices: 8 个样本的索引列表
    """
    random.seed(CONFIG['seed'])

    samples_per_subset = CONFIG['samples_per_subset']
    selected_indices = []

    for subset, count in samples_per_subset.items():
        # 获取该 subset 的所有索引
        subset_indices = [
            i for i in range(len(dataset))
            if dataset[i]['subset'] == subset
        ]

        if len(subset_indices) == 0:
            print(f"[WARN] No samples found for {subset}")
            continue

        # 随机选择
        selected = random.sample(subset_indices, min(count, len(subset_indices)))
        selected_indices.extend(selected)

        print(f"  {subset}: selected {len(selected)} samples")

    return selected_indices


def save_sample_list(indices: list, dataset: HazeDensityDataset, output_file: Path):
    """保存样本列表"""
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("Overfit 8 Samples List\n")
        f.write("=" * 60 + "\n")
        f.write(f"Seed: {CONFIG['seed']}\n")
        f.write(f"Total: {len(indices)} samples\n\n")

        for i, idx in enumerate(indices, 1):
            sample = dataset[idx]
            f.write(f"{i}. {sample['subset']}: {sample['filename']}\n")
            f.write(f"   Path: {sample['path']}\n\n")

    print(f"Saved sample list: {output_file}")


def create_visualization(
    images: torch.Tensor,
    targets: torch.Tensor,
    predictions: torch.Tensor,
    epoch: int,
    output_dir: Path,
):
    """
    创建可视化图像

    布局：[Hazy, Target, Prediction, Error] x batch_size
    """
    from PIL import Image
    import torchvision.utils as vutils
    import numpy as np

    # 计算绝对误差
    errors = torch.abs(targets - predictions)

    # 归一化 error 到 [0, 1] (基于 batch 内最大误差)
    error_max = errors.max()
    if error_max > 0:
        errors = errors / error_max

    # 拼接：[Hazy, Target, Prediction, Error]
    # images: [B, 3, H, W]
    # targets: [B, 1, H, W] -> [B, 3, H, W]
    # predictions: [B, 1, H, W] -> [B, 3, H, W]
    # errors: [B, 1, H, W] -> [B, 3, H, W]

    targets_3ch = targets.repeat(1, 3, 1, 1)
    predictions_3ch = predictions.repeat(1, 3, 1, 1)
    errors_3ch = errors.repeat(1, 3, 1, 1)

    # 水平拼接
    batch_visuals = []
    for b in range(images.shape[0]):
        row = torch.cat([
            images[b],
            targets_3ch[b],
            predictions_3ch[b],
            errors_3ch[b],
        ], dim=1)  # [3, W*4, H]
        batch_visuals.append(row)

    grid = torch.stack(batch_visuals, dim=0)  # [B, 3, W*4, H]

    # 转为 PIL
    numpy_image = grid.permute(0, 2, 3, 1).mul(255).clamp(0, 255).byte().cpu().numpy()

    # 保存每张
    for b in range(min(2, len(numpy_image))):
        img = Image.fromarray(numpy_image[b].astype('uint8'), mode='RGB')
        output_file = output_dir / f'epoch_{epoch:03d}_batch{b}.png'
        img.save(output_file, quality=95)

    print(f"  Saved visualization: epoch_{epoch:03d}_batch*.png")


def train_overfit():
    """主训练函数"""
    print_separator("Stage 5C: HazeDensityNet 8-Image Overfit Test")

    # 设置随机种子
    set_seed(CONFIG['seed'])

    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if device.type == "cuda":
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    # 创建输出目录
    checkpoint_dir = Path(CONFIG['checkpoint_dir'])
    result_dir = Path(CONFIG['result_dir'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    # 加载数据集
    print("\nLoading dataset...")
    dataset = HazeDensityDataset(
        root=CONFIG['dataset_root'],
        split='train',
        image_size=CONFIG['image_size'],
        return_clean=False,
        split_file=CONFIG['split_file'],
    )
    print(f"Full train dataset: {len(dataset)} samples")

    # 选择 8 张样本
    print("\nSelecting 8 samples...")
    selected_indices = select_8_samples(dataset)
    print(f"Total selected: {len(selected_indices)} samples")

    # 保存样本列表
    sample_list_file = result_dir / 'sample_list.txt'
    save_sample_list(selected_indices, dataset, sample_list_file)

    # 创建 Subset 和 DataLoader
    subset = Subset(dataset, selected_indices)
    loader = DataLoader(
        subset,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        num_workers=0,
        pin_memory=True if device.type == "cuda" else False,
    )
    print(f"DataLoader: {len(loader)} batches")

    # 创建模型
    print("\nCreating model...")
    model = HazeDensityNet(base_channels=32, use_sigmoid=True).to(device)
    model_summary = get_model_summary(model)
    print(model_summary)

    # 保存模型信息
    model_info_file = result_dir / 'model_info.txt'
    with open(model_info_file, 'w', encoding='utf-8') as f:
        f.write(model_summary)
    print(f"Saved model info: {model_info_file}")

    # 创建 Physical Prior
    print("\nCreating Physical Prior...")
    physical_prior = PhysicalPriorModule().to(device)
    physical_prior.eval()
    for param in physical_prior.parameters():
        param.requires_grad = False

    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['lr'])

    # Loss
    criterion = nn.MSELoss()

    # AMP
    use_amp = CONFIG['amp'] and device.type == "cuda"
    scaler = GradScaler() if use_amp else None

    # 训练记录
    train_log = []
    best_loss = float('inf')

    # 重置 GPU 内存统计
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    print_separator("Training Start")

    start_time = time.time()

    for epoch in range(1, CONFIG['epochs'] + 1):
        model.train()
        epoch_loss = 0.0
        epoch_steps = 0

        for batch_idx, batch in enumerate(loader):
            # 加载数据
            images = batch['image'].to(device, non_blocking=True)

            # 计算 target (no_grad)
            with torch.no_grad():
                targets = physical_prior(images)

            # 前向
            if use_amp:
                with autocast():
                    predictions = model(images)
                    loss = criterion(predictions, targets)

                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                predictions = model(images)
                loss = criterion(predictions, targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            epoch_loss += loss.item()
            epoch_steps += 1

        avg_loss = epoch_loss / epoch_steps
        elapsed_time = time.time() - start_time

        # 记录
        log_entry = {
            'epoch': epoch,
            'loss': avg_loss,
            'lr': CONFIG['lr'],
            'time': elapsed_time,
        }
        train_log.append(log_entry)

        # 更新 best
        if avg_loss < best_loss:
            best_loss = avg_loss

        # 打印
        print(f"Epoch {epoch:2d}/{CONFIG['epochs']}: "
              f"loss={avg_loss:.6f}, "
              f"best={best_loss:.6f}, "
              f"time={elapsed_time:.1f}s")

        # 保存 checkpoint
        if epoch % 10 == 0 or epoch == 1 or epoch == CONFIG['epochs']:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_loss': best_loss,
                'config': CONFIG,
            }

            # latest
            torch.save(checkpoint, checkpoint_dir / 'latest.pth')

            # best
            if avg_loss == best_loss:
                torch.save(checkpoint, checkpoint_dir / 'best.pth')

            # epoch
            torch.save(checkpoint, checkpoint_dir / f'epoch_{epoch:03d}.pth')

        # 可视化
        if epoch in VIS_EPOCHS:
            # 评估模式，可视化
            model.eval()
            with torch.no_grad():
                for batch in loader:
                    images = batch['image'].to(device, non_blocking=True)
                    targets = physical_prior(images)
                    predictions = model(images)

                    create_visualization(
                        images.cpu(),
                        targets.cpu(),
                        predictions.cpu(),
                        epoch,
                        result_dir,
                    )
                    break  # 只可视化第一个 batch

            model.train()

    # 训练结束
    total_time = time.time() - start_time

    # GPU 内存
    peak_memory = 0
    if device.type == "cuda":
        peak_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)

    # 保存训练日志
    log_file = result_dir / 'train_loss.csv'
    with open(log_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['epoch', 'loss', 'lr', 'time'])
        writer.writeheader()
        writer.writerows(train_log)
    print(f"\nSaved training log: {log_file}")

    # 打印总结
    print_separator("Training Complete")

    initial_loss = train_log[0]['loss']
    final_loss = train_log[-1]['loss']
    loss_reduction = (initial_loss - final_loss) / initial_loss * 100

    print(f"\nTraining Summary:")
    print(f"  Initial loss: {initial_loss:.6f}")
    print(f"  Final loss:   {final_loss:.6f}")
    print(f"  Best loss:    {best_loss:.6f}")
    print(f"  Loss reduction: {loss_reduction:.2f}%")
    print(f"  Total time: {total_time:.1f}s ({total_time/CONFIG['epochs']:.2f}s/epoch)")
    print(f"  Peak GPU memory: {peak_memory:.1f} MB")

    # 保存总结
    summary_file = result_dir / 'training_summary.txt'
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("Stage 5C: 8-Image Overfit Training Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Initial loss: {initial_loss:.6f}\n")
        f.write(f"Final loss: {final_loss:.6f}\n")
        f.write(f"Best loss: {best_loss:.6f}\n")
        f.write(f"Loss reduction: {loss_reduction:.2f}%\n")
        f.write(f"Total time: {total_time:.1f}s\n")
        f.write(f"Time per epoch: {total_time/CONFIG['epochs']:.2f}s\n")
        f.write(f"Peak GPU memory: {peak_memory:.1f} MB\n")
        f.write(f"Device: {device}\n")
        f.write(f"\nConfig:\n")
        for k, v in CONFIG.items():
            f.write(f"  {k}: {v}\n")

    print(f"Saved summary: {summary_file}")

    # 验收检查
    print_separator("Acceptance Check")

    all_pass = True

    # 1. No NaN
    if not any(torch.isnan(torch.tensor(l['loss'])) for l in train_log):
        print("[OK] No NaN in training")
    else:
        print("[FAIL] NaN detected in training")
        all_pass = False

    # 2. No Inf
    if not any(torch.isinf(torch.tensor(l['loss'])) for l in train_log):
        print("[OK] No Inf in training")
    else:
        print("[FAIL] Inf detected in training")
        all_pass = False

    # 3. Loss decreases
    if final_loss < initial_loss:
        print(f"[OK] Loss decreased ({initial_loss:.6f} -> {final_loss:.6f})")
    else:
        print(f"[WARN] Loss did not decrease ({initial_loss:.6f} -> {final_loss:.6f})")
        all_pass = False

    # 4. Checkpoint saved
    if (checkpoint_dir / 'latest.pth').exists():
        print("[OK] Checkpoint saved")
    else:
        print("[FAIL] Checkpoint not saved")
        all_pass = False

    # 5. Visualization saved
    vis_files = list(result_dir.glob('epoch_*.png'))
    if len(vis_files) > 0:
        print(f"[OK] Visualization saved ({len(vis_files)} files)")
    else:
        print("[WARN] No visualization saved")

    print_separator("Result")

    if all_pass:
        print("[OK] Stage 5C Overfit Test PASSED")
    else:
        print("[WARN] Stage 5C Overfit Test has issues")

    return all_pass


def get_model_summary(model: HazeDensityNet) -> str:
    """获取模型摘要"""
    stats = model.get_parameter_stats()

    summary = []
    summary.append("=" * 60)
    summary.append("HazeDensityNet Model Summary")
    summary.append("=" * 60)
    summary.append(f"base_channels: {model.base_channels}")
    summary.append(f"use_sigmoid: {model.use_sigmoid}")
    summary.append("")
    summary.append("Parameter Statistics:")
    summary.append(f"  Encoder:    {stats['encoder']:,} ({stats['encoder']/stats['total']*100:.1f}%)")
    summary.append(f"  MultiScale: {stats['multiscale']:,} ({stats['multiscale']/stats['total']*100:.1f}%)")
    summary.append(f"  Fusion:     {stats['fusion']:,} ({stats['fusion']/stats['total']*100:.1f}%)")
    summary.append(f"  Decoder:    {stats['decoder']:,} ({stats['decoder']/stats['total']*100:.1f}%)")
    summary.append("-" * 60)
    summary.append(f"  Total:      {stats['total']:,}")
    summary.append("=" * 60)

    return "\n".join(summary)


def main():
    """主函数"""
    try:
        success = train_overfit()
        return success
    except Exception as e:
        print(f"\n[ERROR] Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
