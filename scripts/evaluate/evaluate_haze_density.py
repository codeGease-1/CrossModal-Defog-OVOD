#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stage 5E: HazeDensityNet Test Evaluation

在 RSHaze+ 官方 test split 上进行完整评估。

功能:
    1. 加载 best checkpoint
    2. 在 test set 上计算指标
    3. 分 subset 统计 (RSHaze_G/L/S)
    4. Prediction distribution audit
    5. 生成可视化

使用方法:
    python scripts/evaluate_haze_density.py
"""

import sys
from pathlib import Path
import argparse
import random

# 设置路径
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from collections import defaultdict

from src.data import build_rshazeplus_dataloader
from src.models.haze_density import HazeDensityNet
from src.models.haze_density.physical_prior import PhysicalPriorModule


def print_separator(title: str):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def parse_args():
    parser = argparse.ArgumentParser(description="HazeDensityNet Test Evaluation")
    parser.add_argument('--checkpoint', type=str,
                        default='experiments/haze_density/checkpoints/formal/best.pth',
                        help='Checkpoint 路径')
    parser.add_argument('--dataset_root', type=str, default='datasets/RSHaze+',
                        help='数据集根目录')
    parser.add_argument('--split_file', type=str, default='experiments/haze_density/rshazeplus_split.json',
                        help='split 文件路径')
    parser.add_argument('--image_size', type=int, default=256,
                        help='图像尺寸')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='batch size')
    parser.add_argument('--num_samples_per_subset', type=int, default=16,
                        help='每个 subset 可视化样本数')
    return parser.parse_args()


def compute_metrics(predictions, targets):
    """
    计算评估指标

    Args:
        predictions: [N, 1, H, W]
        targets: [N, 1, H, W]

    Returns:
        dict with MSE, MAE, RMSE, Pearson correlation
    """
    pred_flat = predictions.flatten()
    target_flat = targets.flatten()

    # MSE
    mse = torch.mean((pred_flat - target_flat) ** 2).item()

    # MAE
    mae = torch.mean(torch.abs(pred_flat - target_flat)).item()

    # RMSE (使用 math.sqrt 因为 mse 已经是 float)
    import math
    rmse = math.sqrt(mse)

    # Pearson correlation
    corr = torch.corrcoef(torch.stack([pred_flat, target_flat]))[0, 1].item()

    return {
        'MSE': mse,
        'MAE': mae,
        'RMSE': rmse,
        'Pearson': corr,
    }


def compute_distribution_stats(tensor: torch.Tensor) -> dict:
    """计算分布统计"""
    tensor_np = tensor.cpu().float().numpy()

    return {
        'mean': float(np.mean(tensor_np)),
        'std': float(np.std(tensor_np)),
        'min': float(np.min(tensor_np)),
        'max': float(np.max(tensor_np)),
        'p5': float(np.percentile(tensor_np, 5)),
        'p25': float(np.percentile(tensor_np, 25)),
        'p50': float(np.percentile(tensor_np, 50)),
        'p75': float(np.percentile(tensor_np, 75)),
        'p95': float(np.percentile(tensor_np, 95)),
    }


def evaluate_test_set(args):
    """主评估函数"""
    print_separator("Stage 5E: HazeDensityNet Test Evaluation")

    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # 加载 checkpoint
    print(f"\nLoading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')

    config = checkpoint.get('config', {})
    checkpoint_epoch = checkpoint.get('epoch', 'unknown')
    checkpoint_val_loss = checkpoint.get('val_loss', 'unknown')

    print(f"  Epoch: {checkpoint_epoch}")
    print(f"  Val Loss: {checkpoint_val_loss:.6f}")

    # 创建模型
    print("\nCreating model...")
    base_channels = config.get('base_channels', 32)
    use_sigmoid = config.get('use_sigmoid', True)

    model = HazeDensityNet(base_channels=base_channels, use_sigmoid=use_sigmoid).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"  base_channels: {base_channels}")
    print(f"  use_sigmoid: {use_sigmoid}")

    # 创建 Physical Prior
    print("\nCreating Physical Prior...")
    physical_prior = PhysicalPriorModule().to(device)
    physical_prior.eval()

    # 加载 test 数据集
    print("\nLoading test dataset...")
    test_loader = build_rshazeplus_dataloader(
        root=args.dataset_root,
        split='test',
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=0,
        pin_memory=False,
        split_file=args.split_file,
    )
    print(f"Test loader: {len(test_loader)} batches ({len(test_loader.dataset)} samples)")

    # 获取数据集统计
    dataset_stats = test_loader.dataset.get_stats()
    print(f"\nTest dataset statistics:")
    print(f"  Total samples: {dataset_stats['total_samples']}")
    print(f"  Subset counts: {dataset_stats['current_subset_counts']}")

    # ========== 评估 ==========
    print_separator("Evaluation")

    all_predictions = []
    all_targets = []
    all_filenames = []
    all_subsets = []

    # 按 subset 分组
    subset_predictions = defaultdict(list)
    subset_targets = defaultdict(list)
    subset_filenames = defaultdict(list)

    with torch.no_grad():
        for batch in test_loader:
            images = batch['image'].to(device, non_blocking=True)
            filenames = batch['filename']
            subsets = batch['subset']

            # Target
            targets = physical_prior(images)

            # Prediction
            predictions = model(images)

            # 收集
            all_predictions.append(predictions.cpu())
            all_targets.append(targets.cpu())
            all_filenames.extend(filenames)
            all_subsets.extend(subsets)

            # 按 subset 分组
            for i, subset in enumerate(subsets):
                subset_predictions[subset].append(predictions[i:i+1].cpu())
                subset_targets[subset].append(targets[i:i+1].cpu())
                subset_filenames[subset].append(filenames[i])

    # 拼接所有 batch
    predictions_all = torch.cat(all_predictions, dim=0)
    targets_all = torch.cat(all_targets, dim=0)

    print(f"\nTotal test samples: {predictions_all.shape[0]}")
    print(f"Shape per sample: {predictions_all.shape[1:]}")

    # 检查 NaN
    if torch.isnan(predictions_all).any():
        print("[ERROR] NaN detected in predictions!")
        return
    if torch.isnan(targets_all).any():
        print("[ERROR] NaN detected in targets!")
        return
    print("[OK] No NaN detected")

    # 检查范围
    pred_min, pred_max = predictions_all.min().item(), predictions_all.max().item()
    print(f"Prediction range: [{pred_min:.4f}, {pred_max:.4f}]")

    if pred_min < 0 or pred_max > 1:
        print(f"[WARN] Prediction range outside [0, 1]!")
    else:
        print("[OK] Prediction range within [0, 1]")

    # ========== 整体指标 ==========
    print_separator("Overall Metrics")

    overall_metrics = compute_metrics(predictions_all, targets_all)

    print("\nOverall Test Metrics:")
    print(f"  MSE:     {overall_metrics['MSE']:.6f}")
    print(f"  MAE:     {overall_metrics['MAE']:.6f}")
    print(f"  RMSE:    {overall_metrics['RMSE']:.6f}")
    print(f"  Pearson: {overall_metrics['Pearson']:.6f}")

    # ========== 分 Subset 指标 ==========
    print_separator("Metrics by Subset")

    subset_metrics = {}
    for subset in ['RSHaze_G', 'RSHaze_L', 'RSHaze_S']:
        if subset not in subset_predictions or len(subset_predictions[subset]) == 0:
            print(f"\n{subset}: No samples")
            continue

        subset_pred = torch.cat(subset_predictions[subset], dim=0)
        subset_target = torch.cat(subset_targets[subset], dim=0)

        metrics = compute_metrics(subset_pred, subset_target)
        subset_metrics[subset] = metrics

        print(f"\n{subset} ({len(subset_predictions[subset])} samples):")
        print(f"  MAE:     {metrics['MAE']:.6f}")
        print(f"  RMSE:    {metrics['RMSE']:.6f}")
        print(f"  Pearson: {metrics['Pearson']:.6f}")

    # ========== Prediction Distribution Audit ==========
    print_separator("Prediction Distribution Audit")

    pred_stats = compute_distribution_stats(predictions_all)
    target_stats = compute_distribution_stats(targets_all)

    print("\nPrediction Distribution:")
    for k, v in pred_stats.items():
        print(f"  {k}: {v:.6f}")

    print("\nTarget Distribution:")
    for k, v in target_stats.items():
        print(f"  {k}: {v:.6f}")

    # ========== Visualization ==========
    print_separator("Generating Visualization")

    output_dir = Path('experiments/haze_density/results/test_evaluation')
    output_dir.mkdir(parents=True, exist_ok=True)

    random.seed(42)

    # 重新运行一次以获取图像（用于可视化）
    with torch.no_grad():
        for batch in test_loader:
            images = batch['image'].to(device, non_blocking=True)
            filenames = batch['filename']
            subsets = batch['subset']

            # Target
            targets = physical_prior(images)

            # Prediction
            predictions = model(images)

            # 可视化
            for i, subset in enumerate(subsets):
                filename = filenames[i]

                # 检查是否已保存足够样本
                existing_files = list(output_dir.glob(f'{subset}_*.png'))
                if len(existing_files) >= args.num_samples_per_subset:
                    continue

                image = images[i:i+1]
                pred = predictions[i:i+1]
                target = targets[i:i+1]

                # Compute error
                error = torch.abs(target - pred)
                error_max = error.max().item()
                if error_max > 0:
                    error = error / error_max

                # Convert to 3-channel
                image_3ch = image[0]
                target_3ch = target.repeat(1, 3, 1, 1)[0]
                pred_3ch = pred.repeat(1, 3, 1, 1)[0]
                error_3ch = error.repeat(1, 3, 1, 1)[0]

                # Create grid: [Hazy, Target, Prediction, Error]
                row = torch.cat([image_3ch, target_3ch, pred_3ch, error_3ch], dim=1)

                # Save
                numpy_image = row.permute(1, 2, 0).mul(255).clamp(0, 255).byte().cpu().numpy()
                pil_image = Image.fromarray(numpy_image.astype('uint8'), mode='RGB')

                output_file = output_dir / f'{subset}_{Path(filename).stem}.png'
                pil_image.save(output_file, quality=95)

    # 统计保存的可视化数量
    for subset in ['RSHaze_G', 'RSHaze_L', 'RSHaze_S']:
        count = len(list(output_dir.glob(f'{subset}_*.png')))
        print(f"  {subset}: Saved {count} visualizations")

    # ========== 保存报告 ==========
    print_separator("Saving Report")

    report_file = output_dir / 'test_metrics.txt'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("Stage 5E: HazeDensityNet Test Evaluation Report\n")
        f.write("=" * 60 + "\n\n")

        f.write("Checkpoint Information:\n")
        f.write(f"  Path: {args.checkpoint}\n")
        f.write(f"  Epoch: {checkpoint_epoch}\n")
        f.write(f"  Val Loss: {checkpoint_val_loss:.6f}\n\n")

        f.write("Dataset Information:\n")
        f.write(f"  Total test samples: {len(test_loader.dataset)}\n")
        f.write(f"  Subset counts: {dataset_stats['current_subset_counts']}\n\n")

        f.write("Overall Metrics:\n")
        for k, v in overall_metrics.items():
            f.write(f"  {k}: {v:.6f}\n")
        f.write("\n")

        f.write("Metrics by Subset:\n")
        for subset, metrics in subset_metrics.items():
            f.write(f"\n{subset}:\n")
            for k, v in metrics.items():
                f.write(f"  {k}: {v:.6f}\n")
        f.write("\n")

        f.write("Prediction Distribution:\n")
        for k, v in pred_stats.items():
            f.write(f"  {k}: {v:.6f}\n")
        f.write("\n")

        f.write("Target Distribution:\n")
        for k, v in target_stats.items():
            f.write(f"  {k}: {v:.6f}\n")

    print(f"Saved report: {report_file}")

    # ========== 验收检查 ==========
    print_separator("Acceptance Check")

    checks_passed = True

    # 检查 1: 无 NaN
    if torch.isnan(predictions_all).any():
        print("[FAIL] NaN detected in predictions")
        checks_passed = False
    else:
        print("[PASS] No NaN in predictions")

    # 检查 2: Prediction 范围 [0, 1]
    if pred_min >= 0 and pred_max <= 1:
        print(f"[PASS] Prediction range [{pred_min:.4f}, {pred_max:.4f}] within [0, 1]")
    else:
        print(f"[FAIL] Prediction range [{pred_min:.4f}, {pred_max:.4f}] outside [0, 1]")
        checks_passed = False

    # 检查 3: 三个 subset 均完成
    required_subsets = {'RSHaze_G', 'RSHaze_L', 'RSHaze_S'}
    completed_subsets = set(subset_metrics.keys())
    if required_subsets.issubset(completed_subsets):
        print(f"[PASS] All subsets completed: {completed_subsets}")
    else:
        missing = required_subsets - completed_subsets
        print(f"[FAIL] Missing subsets: {missing}")
        checks_passed = False

    # 检查 4: Metrics 保存
    if report_file.exists():
        print(f"[PASS] Metrics saved to {report_file}")
    else:
        print(f"[FAIL] Metrics not saved")
        checks_passed = False

    if checks_passed:
        print("\n[ALL CHECKS PASSED]")
    else:
        print("\n[SOME CHECKS FAILED]")

    return checks_passed


if __name__ == "__main__":
    args = parse_args()
    success = evaluate_test_set(args)
    sys.exit(0 if success else 1)
