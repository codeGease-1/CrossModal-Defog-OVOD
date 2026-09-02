#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
雾密度感知网络正式训练脚本 (Stage 5D)

训练流程:
    1. 加载配置
    2. 初始化模型、优化器
    3. 加载 train/val 数据集
    4. 训练循环:
        - 计算 Physical Prior (no_grad)
        - 前向传播
        - 计算 loss
        - 反向传播
        - 验证
        - 保存 checkpoint
    5. 支持断点续训

使用方法:
    # 基本训练 (5 epochs smoke test)
    python scripts/train_haze_density.py --epochs 5

    # 完整训练 (50 epochs)
    python scripts/train_haze_density.py --epochs 50

    # 断点续训
    python scripts/train_haze_density.py --resume experiments/haze_density/checkpoints/latest.pth

    # 自定义参数
    python scripts/train_haze_density.py --batch_size 8 --lr 5e-4
"""

import sys
import os
from pathlib import Path
import time
import random
import csv
import argparse

# 设置路径
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

from src.data import HazeDensityDataset, build_rshazeplus_dataloader
from src.models.haze_density import HazeDensityNet
from src.models.haze_density.physical_prior import PhysicalPriorModule
from src.utils.path_utils import get_dataset_root, get_split_file_path, get_checkpoint_dir, get_result_dir


# ============================================================================
# 默认配置
# ============================================================================

DEFAULT_CONFIG = {
    # 数据配置
    'dataset_root': 'datasets/RSHaze+',
    'split_file': 'experiments/haze_density/rshazeplus_split.json',
    'image_size': 256,
    'batch_size': 4,
    'num_workers': 2,
    'pin_memory': True,

    # 模型配置
    'base_channels': 32,
    'use_sigmoid': True,

    # 训练配置
    'epochs': 50,
    'lr': 1e-4,
    'amp': True,
    'seed': 42,

    # 路径配置
    'checkpoint_dir': 'experiments/haze_density/checkpoints/formal',
    'result_dir': 'experiments/haze_density/results/formal',

    # Physical Prior 配置
    'window_size': 15,
    'guided_radius': 15,
    'guided_eps': 0.01,
}


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


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="HazeDensityNet Formal Training")

    # 训练参数
    parser.add_argument('--epochs', type=int, default=DEFAULT_CONFIG['epochs'],
                        help='训练轮数 (默认：50)')
    parser.add_argument('--batch_size', type=int, default=DEFAULT_CONFIG['batch_size'],
                        help='batch size (默认：4)')
    parser.add_argument('--lr', type=float, default=DEFAULT_CONFIG['lr'],
                        help='学习率 (默认：1e-4)')
    parser.add_argument('--image_size', type=int, default=DEFAULT_CONFIG['image_size'],
                        help='图像尺寸 (默认：256)')

    # 路径参数
    parser.add_argument('--dataset_root', type=str, default=None,
                        help='数据集根目录 (默认：自动检测)')
    parser.add_argument('--resume', type=str, default=None,
                        help='从 checkpoint 恢复训练')
    parser.add_argument('--checkpoint_dir', type=str, default=None,
                        help='checkpoint 保存目录 (默认：自动检测)')
    parser.add_argument('--result_dir', type=str, default=None,
                        help='结果保存目录 (默认：自动检测)')
    parser.add_argument('--force_env', type=str, default=None,
                        help='强制指定环境 (colab/kaggle/local)')

    # 其他参数
    parser.add_argument('--num_workers', type=int, default=DEFAULT_CONFIG['num_workers'],
                        help='DataLoader worker 数 (默认：2)')
    parser.add_argument('--no_amp', action='store_true',
                        help='禁用 AMP 混合精度训练')

    return parser.parse_args()


def load_checkpoint(model, optimizer, checkpoint_path):
    """
    加载 checkpoint

    Returns:
        start_epoch: 起始 epoch
        best_val_loss: 最佳验证 loss
    """
    print(f"\nLoading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    start_epoch = checkpoint['epoch'] + 1
    best_val_loss = checkpoint.get('best_val_loss', float('inf'))

    print(f"  Epoch: {checkpoint['epoch']}")
    print(f"  Best val loss: {best_val_loss:.6f}")
    print(f"  Resuming from epoch {start_epoch}")

    return start_epoch, best_val_loss


def save_checkpoint(model, optimizer, epoch, val_loss, is_best, config, checkpoint_dir):
    """保存 checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'best_val_loss': val_loss if is_best else None,
        'config': config,
    }

    # 保存 latest
    torch.save(checkpoint, checkpoint_dir / 'latest.pth')

    # 保存 best
    if is_best:
        torch.save(checkpoint, checkpoint_dir / 'best.pth')


def validate(model, val_loader, physical_prior, criterion, device, config):
    """
    验证阶段

    Returns:
        val_loss: 平均验证 loss
        pred_stats: prediction 统计
        target_stats: target 统计
    """
    model.eval()

    total_loss = 0.0
    total_samples = 0

    all_pred_min = []
    all_pred_max = []
    all_pred_mean = []
    all_target_min = []
    all_target_max = []
    all_target_mean = []

    with torch.no_grad():
        for batch in val_loader:
            images = batch['image'].to(device, non_blocking=True)

            # 计算 target
            targets = physical_prior(images)

            # 前向
            predictions = model(images)

            # Loss
            loss = criterion(predictions, targets)

            total_loss += loss.item() * images.shape[0]
            total_samples += images.shape[0]

            # 统计
            all_pred_min.append(predictions.min().item())
            all_pred_max.append(predictions.max().item())
            all_pred_mean.append(predictions.mean().item())
            all_target_min.append(targets.min().item())
            all_target_max.append(targets.max().item())
            all_target_mean.append(targets.mean().item())

    avg_loss = total_loss / total_samples if total_samples > 0 else 0

    pred_stats = {
        'min': min(all_pred_min),
        'max': max(all_pred_max),
        'mean': sum(all_pred_mean) / len(all_pred_mean),
    }

    target_stats = {
        'min': min(all_target_min),
        'max': max(all_target_max),
        'mean': sum(all_target_mean) / len(all_target_mean),
    }

    model.train()

    return avg_loss, pred_stats, target_stats


def train_formal(args, config):
    """正式训练主函数"""
    print_separator("Stage 5D: HazeDensityNet Formal Training")

    # 设置随机种子
    set_seed(config['seed'])

    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if device.type == "cuda":
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    # 自动检测路径
    dataset_root = get_dataset_root(force_env=args.force_env)
    split_file = get_split_file_path()
    checkpoint_dir = Path(args.checkpoint_dir if args.checkpoint_dir else get_checkpoint_dir())
    result_dir = Path(args.result_dir if args.result_dir else get_result_dir())

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nDataset root: {dataset_root}")
    print(f"Split file: {split_file}")
    print(f"Checkpoint dir: {checkpoint_dir}")
    print(f"Result dir: {result_dir}")

    # 创建模型
    print("\nCreating model...")
    model = HazeDensityNet(
        base_channels=config['base_channels'],
        use_sigmoid=config['use_sigmoid'],
    ).to(device)

    model_summary = get_model_summary(model)
    print(model_summary)

    # 保存模型信息
    with open(result_dir / 'model_info.txt', 'w', encoding='utf-8') as f:
        f.write(model_summary)
        f.write(f"\nDevice: {device}\n")

    # 创建 Physical Prior
    print("\nCreating Physical Prior...")
    physical_prior = PhysicalPriorModule(
        window_size=config['window_size'],
        guided_radius=config['guided_radius'],
        guided_eps=config['guided_eps'],
    ).to(device)
    physical_prior.eval()
    for param in physical_prior.parameters():
        param.requires_grad = False

    # 创建 DataLoader
    print("\nLoading datasets...")

    train_loader = build_rshazeplus_dataloader(
        root=dataset_root,
        split='train',
        image_size=config['image_size'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        pin_memory=config['pin_memory'] and device.type == "cuda",
        split_file=split_file,
    )

    val_loader = build_rshazeplus_dataloader(
        root=dataset_root,
        split='val',
        image_size=config['image_size'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        pin_memory=config['pin_memory'] and device.type == "cuda",
        split_file=split_file,
    )

    print(f"Train loader: {len(train_loader)} batches ({len(train_loader.dataset)} samples)")
    print(f"Val loader: {len(val_loader)} batches ({len(val_loader.dataset)} samples)")

    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])

    # Loss
    criterion = nn.MSELoss()

    # AMP
    use_amp = config['amp'] and device.type == "cuda"
    scaler = GradScaler() if use_amp else None
    print(f"AMP: {'Enabled' if use_amp else 'Disabled'}")

    # 训练状态
    start_epoch = 1
    best_val_loss = float('inf')

    # 尝试恢复训练
    if config.get('resume'):
        start_epoch, best_val_loss = load_checkpoint(
            model, optimizer, config['resume']
        )

    # 重置 GPU 内存统计
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # 训练日志
    train_log = []

    print_separator("Training Start")

    for epoch in range(start_epoch, config['epochs'] + 1):
        epoch_start = time.time()

        # ========== Training ==========
        model.train()
        total_loss = 0.0
        total_samples = 0

        for batch_idx, batch in enumerate(train_loader):
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

            total_loss += loss.item() * images.shape[0]
            total_samples += images.shape[0]

        avg_train_loss = total_loss / total_samples if total_samples > 0 else 0

        # ========== Validation ==========
        val_loss, pred_stats, target_stats = validate(
            model, val_loader, physical_prior, criterion, device, config
        )

        # ========== 更新 best ==========
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss

        # ========== 保存 checkpoint ==========
        save_checkpoint(
            model, optimizer, epoch, val_loss, is_best, config, checkpoint_dir
        )

        # ========== 计算时间 ==========
        epoch_time = time.time() - epoch_start

        # ========== 打印日志 ==========
        print(f"Epoch {epoch:2d}/{config['epochs']}: "
              f"train_loss={avg_train_loss:.6f}, "
              f"val_loss={val_loss:.6f}, "
              f"best={best_val_loss:.6f}, "
              f"time={epoch_time:.1f}s")

        # ========== 记录日志 ==========
        log_entry = {
            'epoch': epoch,
            'train_loss': avg_train_loss,
            'val_loss': val_loss,
            'lr': config['lr'],
            'epoch_time': epoch_time,
            'pred_min': pred_stats['min'],
            'pred_max': pred_stats['max'],
            'pred_mean': pred_stats['mean'],
            'target_min': target_stats['min'],
            'target_max': target_stats['max'],
            'target_mean': target_stats['mean'],
        }
        train_log.append(log_entry)

    # ========== 训练结束 ==========
    total_time = train_log[-1]['epoch_time'] * config['epochs'] if train_log else 0

    # GPU 内存
    peak_memory = 0
    if device.type == "cuda":
        peak_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)

    # 保存训练日志
    log_file = result_dir / 'train_log.csv'
    with open(log_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=log_entry.keys())
        writer.writeheader()
        writer.writerows(train_log)
    print(f"\nSaved training log: {log_file}")

    # 保存总结
    summary_file = result_dir / 'training_summary.txt'
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("Stage 5D: Formal Training Summary\n")
        f.write("=" * 60 + "\n\n")

        if train_log:
            f.write(f"Initial train loss: {train_log[0]['train_loss']:.6f}\n")
            f.write(f"Final train loss: {train_log[-1]['train_loss']:.6f}\n")
            f.write(f"Initial val loss: {train_log[0]['val_loss']:.6f}\n")
            f.write(f"Final val loss: {train_log[-1]['val_loss']:.6f}\n")
            f.write(f"Best val loss: {best_val_loss:.6f}\n")
            f.write(f"Total epochs: {config['epochs']}\n")
            f.write(f"Total time: {total_time:.1f}s\n")
            f.write(f"Time per epoch: {total_time/config['epochs']:.2f}s\n")

        f.write(f"Peak GPU memory: {peak_memory:.1f} MB\n")
        f.write(f"Device: {device}\n")
        f.write(f"\nConfig:\n")
        for k, v in config.items():
            if k != 'resume':
                f.write(f"  {k}: {v}\n")

    print(f"Saved summary: {summary_file}")

    # ========== 验收检查 ==========
    print_separator("Acceptance Check")

    all_pass = True

    # 1. No NaN
    if not any(log['train_loss'] != log['train_loss'] for log in train_log):  # NaN check
        print("[OK] No NaN in training")
    else:
        print("[FAIL] NaN detected in training")
        all_pass = False

    # 2. No Inf
    if not any(abs(log['train_loss']) == float('inf') for log in train_log):
        print("[OK] No Inf in training")
    else:
        print("[FAIL] Inf detected in training")
        all_pass = False

    # 3. Prediction range
    final_pred = train_log[-1]
    if 0 <= final_pred['pred_min'] and final_pred['pred_max'] <= 1:
        print(f"[OK] Prediction range: [{final_pred['pred_min']:.4f}, {final_pred['pred_max']:.4f}]")
    else:
        print(f"[WARN] Prediction range out of [0,1]: [{final_pred['pred_min']:.4f}, {final_pred['pred_max']:.4f}]")

    # 4. Target range
    if 0 <= final_pred['target_min'] and final_pred['target_max'] <= 1:
        print(f"[OK] Target range: [{final_pred['target_min']:.4f}, {final_pred['target_max']:.4f}]")
    else:
        print(f"[WARN] Target range out of [0,1]: [{final_pred['target_min']:.4f}, {final_pred['target_max']:.4f}]")

    # 5. Checkpoint saved
    if (checkpoint_dir / 'latest.pth').exists() and (checkpoint_dir / 'best.pth').exists():
        print("[OK] Checkpoints saved (latest.pth, best.pth)")
    else:
        print("[FAIL] Checkpoints not saved properly")
        all_pass = False

    print_separator("Result")

    if all_pass:
        print("[OK] Stage 5D Formal Training PASSED")
    else:
        print("[WARN] Stage 5D Formal Training has issues")

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
    args = parse_args()

    # 构建配置
    config = DEFAULT_CONFIG.copy()
    config['epochs'] = args.epochs
    config['batch_size'] = args.batch_size
    config['lr'] = args.lr
    config['image_size'] = args.image_size
    config['num_workers'] = args.num_workers
    config['amp'] = not args.no_amp
    config['resume'] = args.resume

    try:
        success = train_formal(args, config)
        return success
    except Exception as e:
        print(f"\n[ERROR] Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
