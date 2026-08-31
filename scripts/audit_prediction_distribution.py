#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stage 5D-1: Prediction Distribution Audit

目标：
1. 完整统计 prediction/target distribution
2. 分析为什么 prediction.min() = 0.5
3. 按低/中/高密度区域分析误差
4. 生成可视化
5. 给出明确结论和修复方案

使用方法:
    python scripts/audit_prediction_distribution.py \
        --checkpoint experiments/haze_density/checkpoints/formal/best.pth
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
import torchvision.utils as vutils

from src.data import build_rshazeplus_dataloader
from src.models.haze_density import HazeDensityNet
from src.models.haze_density.physical_prior import PhysicalPriorModule


def print_separator(title: str):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def parse_args():
    parser = argparse.ArgumentParser(description="Prediction Distribution Audit")
    parser.add_argument('--checkpoint', type=str,
                        default='experiments/haze_density/checkpoints/formal/best.pth',
                        help='Checkpoint 路径')
    parser.add_argument('--image_size', type=int, default=256,
                        help='图像尺寸')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='batch size')
    parser.add_argument('--num_samples', type=int, default=8,
                        help='可视化样本数')
    return parser.parse_args()


def compute_percentile(tensor: torch.Tensor, percentiles: list) -> dict:
    """计算百分位数 - 使用 numpy 避免 torch.quantile 大小限制"""
    # 转换为 numpy 数组
    tensor_np = tensor.cpu().float().numpy()
    result = {}
    for p in percentiles:
        result[f'p{int(p*100)}'] = float(np.percentile(tensor_np, p * 100))
    return result


def compute_below_threshold(tensor: torch.Tensor, thresholds: list) -> dict:
    """计算低于各阈值的像素比例"""
    tensor_flat = tensor.flatten()
    total = tensor_flat.numel()
    result = {}
    for t in thresholds:
        count = (tensor_flat < t).sum().item()
        result[f'<{t}'] = count / total * 100
    return result


def audit_prediction_distribution(args):
    """主审计函数"""
    print_separator("Stage 5D-1: Prediction Distribution Audit")

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

    # 加载 val 数据集
    print("\nLoading validation dataset...")
    val_loader = build_rshazeplus_dataloader(
        root=config.get('dataset_root', 'datasets/RSHaze+'),
        split='val',
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=0,
        split_file=config.get('split_file', 'experiments/haze_density/rshazeplus_split.json'),
    )
    print(f"Val loader: {len(val_loader)} batches ({len(val_loader.dataset)} samples)")

    # ========== 完整统计 ==========
    print("\n" + "-" * 60)
    print("Computing full statistics...")

    all_predictions = []
    all_targets = []
    all_errors = []

    with torch.no_grad():
        for batch in val_loader:
            images = batch['image'].to(device, non_blocking=True)

            # Target
            targets = physical_prior(images)

            # Prediction
            predictions = model(images)

            all_predictions.append(predictions.cpu())
            all_targets.append(targets.cpu())
            all_errors.append(torch.abs(predictions - targets).cpu())

    # 拼接所有 batch
    predictions_all = torch.cat(all_predictions, dim=0)  # [N, 1, H, W]
    targets_all = torch.cat(all_targets, dim=0)
    errors_all = torch.cat(all_errors, dim=0)

    print(f"Total samples: {predictions_all.shape[0]}")
    print(f"Shape per sample: {predictions_all.shape[1:]}")

    # ========== Prediction Distribution ==========
    print_separator("Prediction Distribution")

    pred_stats = {
        'min': float(predictions_all.min().item()),
        'max': float(predictions_all.max().item()),
        'mean': float(predictions_all.mean().item()),
        'std': float(predictions_all.std().item()),
    }
    pred_stats.update(compute_percentile(predictions_all, [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]))
    pred_stats.update(compute_below_threshold(predictions_all, [0.1, 0.2, 0.3, 0.4, 0.5]))

    print("\nPrediction Statistics:")
    for k, v in pred_stats.items():
        if isinstance(v, float) and 'p' not in k and '<' not in k:
            print(f"  {k}: {v:.6f}")
        elif 'p' in k:
            print(f"  {k}: {v:.6f}")
        else:
            print(f"  {k}: {v:.2f}%")

    # ========== Target Distribution ==========
    print_separator("Target (S_final) Distribution")

    target_stats = {
        'min': float(targets_all.min().item()),
        'max': float(targets_all.max().item()),
        'mean': float(targets_all.mean().item()),
        'std': float(targets_all.std().item()),
    }
    target_stats.update(compute_percentile(targets_all, [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]))
    target_stats.update(compute_below_threshold(targets_all, [0.1, 0.2, 0.3, 0.4, 0.5]))

    print("\nTarget Statistics:")
    for k, v in target_stats.items():
        if isinstance(v, float) and 'p' not in k and '<' not in k:
            print(f"  {k}: {v:.6f}")
        elif 'p' in k:
            print(f"  {k}: {v:.6f}")
        else:
            print(f"  {k}: {v:.2f}%")

    # ========== Correlation & Error Metrics ==========
    print_separator("Correlation & Error Metrics")

    # Flatten for correlation
    pred_flat = predictions_all.flatten()
    target_flat = targets_all.flatten()

    # Pearson correlation
    corr = torch.corrcoef(torch.stack([pred_flat, target_flat]))[0, 1].item()
    print(f"\nPearson correlation: {corr:.6f}")

    # MAE
    mae = torch.mean(torch.abs(pred_flat - target_flat)).item()
    print(f"MAE: {mae:.6f}")

    # RMSE
    rmse = torch.sqrt(torch.mean((pred_flat - target_flat) ** 2)).item()
    print(f"RMSE: {rmse:.6f}")

    # MSE
    mse = torch.mean((pred_flat - target_flat) ** 2).item()
    print(f"MSE: {mse:.6f}")

    # ========== Error by Density Region ==========
    print_separator("Error by Density Region")

    # Low density: target < 0.3
    low_mask = targets_all < 0.3
    # Medium density: 0.3 <= target < 0.7
    medium_mask = (targets_all >= 0.3) & (targets_all < 0.7)
    # High density: target >= 0.7
    high_mask = targets_all >= 0.7

    low_pixels = low_mask.sum().item()
    medium_pixels = medium_mask.sum().item()
    high_pixels = high_mask.sum().item()
    total_pixels = low_pixels + medium_pixels + high_pixels

    print(f"\nPixel distribution:")
    print(f"  Low density (<0.3): {low_pixels / total_pixels * 100:.2f}% ({low_pixels:,} pixels)")
    print(f"  Medium density (0.3-0.7): {medium_pixels / total_pixels * 100:.2f}% ({medium_pixels:,} pixels)")
    print(f"  High density (>=0.7): {high_pixels / total_pixels * 100:.2f}% ({high_pixels:,} pixels)")

    # Error by region
    if low_pixels > 0:
        low_mae = torch.mean(errors_all[low_mask]).item()
        low_rmse = torch.sqrt(torch.mean(errors_all[low_mask] ** 2)).item()
        print(f"\nLow density error:")
        print(f"  MAE: {low_mae:.6f}")
        print(f"  RMSE: {low_rmse:.6f}")

    if medium_pixels > 0:
        medium_mae = torch.mean(errors_all[medium_mask]).item()
        medium_rmse = torch.sqrt(torch.mean(errors_all[medium_mask] ** 2)).item()
        print(f"\nMedium density error:")
        print(f"  MAE: {medium_mae:.6f}")
        print(f"  RMSE: {medium_rmse:.6f}")

    if high_pixels > 0:
        high_mae = torch.mean(errors_all[high_mask]).item()
        high_rmse = torch.sqrt(torch.mean(errors_all[high_mask] ** 2)).item()
        print(f"\nHigh density error:")
        print(f"  MAE: {high_mae:.6f}")
        print(f"  RMSE: {high_rmse:.6f}")

    # ========== Visualization ==========
    print_separator("Generating Visualization")

    output_dir = Path('experiments/haze_density/results/formal_prediction_audit')
    output_dir.mkdir(parents=True, exist_ok=True)

    random.seed(42)
    val_indices = list(range(len(val_loader.dataset)))
    selected_indices = random.sample(val_indices, min(args.num_samples, len(val_indices)))

    # 重新加载选中的样本
    from torch.utils.data import Subset, DataLoader
    subset = Subset(val_loader.dataset, selected_indices)
    vis_loader = DataLoader(subset, batch_size=1, shuffle=False)

    vis_count = 0
    for batch in vis_loader:
        if vis_count >= args.num_samples:
            break

        images = batch['image'].to(device)
        filenames = batch['filename']
        subsets = batch['subset']

        with torch.no_grad():
            targets = physical_prior(images)
            predictions = model(images)

        # Compute error
        errors = torch.abs(targets - predictions)
        error_max = errors.max().item()
        if error_max > 0:
            errors = errors / error_max  # Normalize to [0, 1]

        # Convert to 3-channel for visualization
        targets_3ch = targets.repeat(1, 3, 1, 1)
        predictions_3ch = predictions.repeat(1, 3, 1, 1)
        errors_3ch = errors.repeat(1, 3, 1, 1)

        # Create grid: [Hazy, Target, Prediction, Error]
        row = torch.cat([images[0], targets_3ch[0], predictions_3ch[0], errors_3ch[0]], dim=1)

        # Save
        numpy_image = row.permute(1, 2, 0).mul(255).clamp(0, 255).byte().cpu().numpy()
        pil_image = Image.fromarray(numpy_image.astype('uint8'), mode='RGB')

        output_file = output_dir / f'sample_{vis_count:03d}_{subsets[0]}_{Path(filenames[0]).stem}.png'
        pil_image.save(output_file, quality=95)

        print(f"  Saved: {output_file.name}")
        vis_count += 1

    # ========== 问题诊断 ==========
    print_separator("Problem Diagnosis")

    # 检查 decoder 结构
    decoder = model.decoder
    print("\nDecoder structure:")
    print(f"  use_sigmoid: {decoder.use_sigmoid}")

    # 检查是否有 ReLU before Sigmoid
    has_relu_before_sigmoid = hasattr(decoder, 'relu3')
    print(f"  Has relu3 (ReLU before Sigmoid): {has_relu_before_sigmoid}")

    if has_relu_before_sigmoid and decoder.use_sigmoid:
        print("\n[CRITICAL] Found ReLU → Sigmoid pattern!")
        print("  ReLU output range: [0, ∞)")
        print("  Sigmoid([0, ∞)) range: [0.5, 1)")
        print("  This explains why prediction.min() = 0.5000")

    # ========== 结论 ==========
    print_separator("Conclusion")

    pred_min = pred_stats['min']
    pred_max = pred_stats['max']

    print(f"\nPrediction range: [{pred_min:.4f}, {pred_max:.4f}]")
    print(f"Target range: [{target_stats['min']:.4f}, {target_stats['max']:.4f}]")

    if pred_min >= 0.49 and has_relu_before_sigmoid:
        print("\n[CONCLUSION C] Prediction distribution 明显异常")
        print("  原因：Decoder 中存在 ReLU → Sigmoid 结构")
        print("  影响：Prediction 全部 >= 0.5，无法预测低雾密度区域")
        print("  建议：移除 Decoder 最后的 ReLU，直接使用 Sigmoid")
        print("\n  修复方案:")
        print("    decoder.py 第 94 行：删除 self.relu3 = nn.ReLU(inplace=True)")
        print("    decoder.py 第 120 行：删除 x = self.relu3(x)")
        print("    或改为：self.relu3 = nn.Identity()")
        print("\n  修复后需要重新进行 5 epoch smoke training")
        conclusion = "C"
    elif pred_min < 0.1:
        print("\n[CONCLUSION A] Prediction distribution 正常")
        print("  可以继续正式训练")
        conclusion = "A"
    else:
        print("\n[CONCLUSION B] Prediction distribution 存在轻微偏移")
        print("  建议记录并观察")
        conclusion = "B"

    # ========== 保存报告 ==========
    report_file = output_dir / 'audit_report.txt'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("Stage 5D-1: Prediction Distribution Audit Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Epoch: {checkpoint_epoch}\n")
        f.write(f"Val Loss: {checkpoint_val_loss:.6f}\n\n")

        f.write("Prediction Statistics:\n")
        for k, v in pred_stats.items():
            f.write(f"  {k}: {v:.6f}\n")

        f.write("\nTarget Statistics:\n")
        for k, v in target_stats.items():
            f.write(f"  {k}: {v:.6f}\n")

        f.write(f"\nPearson correlation: {corr:.6f}\n")
        f.write(f"MAE: {mae:.6f}\n")
        f.write(f"RMSE: {rmse:.6f}\n")
        f.write(f"MSE: {mse:.6f}\n")

        f.write(f"\nConclusion: {conclusion}\n")

    print(f"\nSaved report: {report_file}")

    return conclusion


if __name__ == "__main__":
    args = parse_args()
    conclusion = audit_prediction_distribution(args)
    print(f"\nFinal Conclusion: {conclusion}")
    sys.exit(0)
