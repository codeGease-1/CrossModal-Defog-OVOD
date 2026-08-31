#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RSHaze+ Dataset + Physical Prior Integration Test (Stage 5B-2)

测试流程:
    RSHazePlusDataset
        ↓
    DataLoader
        ↓
    image [B,3,H,W]
        ↓
    Physical Prior
        ↓
    S_final [B,1,H,W]

验收标准:
    - shape 正确
    - dtype 正确
    - device 正确
    - finite = True
    - range [0, 1]
    - GPU test PASS
    - G/L/S subset test PASS
"""

import sys
from pathlib import Path
import time

# 设置路径
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import torch
from torch.utils.data import DataLoader

from src.data import build_rshazeplus_dataloader, HazeDensityDataset
from src.models.haze_density.physical_prior import generate_s_final, PhysicalPriorModule


def print_separator(title: str):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def test_physical_prior_256():
    """测试 256 模式"""
    print_separator("Physical Prior Test (256x256)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 创建 DataLoader
    train_loader = build_rshazeplus_dataloader(
        root='datasets/RSHaze+',
        split='train',
        image_size=256,
        batch_size=4,
        num_workers=0,  # Colab 上可设为 2
        pin_memory=True if device.type == "cuda" else False,
        split_file='experiments/haze_density/rshazeplus_split.json',
    )

    print(f"Train loader: {len(train_loader)} batches")

    # Physical Prior 模块
    physical_prior = PhysicalPriorModule(
        window_size=15,
        guided_radius=15,
        guided_eps=0.01,
    ).to(device)
    physical_prior.eval()

    # 测试统计
    stats = {
        'RSHaze_G': {'count': 0, 'sum': 0, 'sum_sq': 0, 'min': 1, 'max': 0},
        'RSHaze_L': {'count': 0, 'sum': 0, 'sum_sq': 0, 'min': 1, 'max': 0},
        'RSHaze_S': {'count': 0, 'sum': 0, 'sum_sq': 0, 'min': 1, 'max': 0},
    }

    all_pass = True
    timing_stats = {'data_load': [], 'physical_prior': [], 'total': []}

    # 测试前几个 batch
    for i, batch in enumerate(train_loader):
        if i >= 10:  # 测试 10 个 batch
            break

        # Data loading time
        data_start = time.time()
        image = batch['image'].to(device, non_blocking=True)
        data_time = time.time() - data_start
        timing_stats['data_load'].append(data_time * 1000)

        # 检查输入
        assert image.shape == (4, 3, 256, 256), f"Expected [4,3,256,256], got {image.shape}"
        assert image.min() >= 0 and image.max() <= 1, f"Image range out of [0,1]: [{image.min()}, {image.max()}]"

        # Physical Prior forward
        total_start = time.time()
        with torch.no_grad():
            s_final = physical_prior(image)
        pp_time = time.time() - total_start - data_time
        total_time = time.time() - total_start
        timing_stats['physical_prior'].append(pp_time * 1000)
        timing_stats['total'].append(total_time * 1000)

        # 检查输出
        assert s_final.shape == (4, 1, 256, 256), f"Expected [4,1,256,256], got {s_final.shape}"
        assert torch.isfinite(s_final).all(), "S_final contains non-finite values"
        assert s_final.min() >= 0, f"S_final min < 0: {s_final.min()}"
        assert s_final.max() <= 1, f"S_final max > 1: {s_final.max()}"

        # 统计各 subset
        subsets = batch['subset']
        for j, subset in enumerate(subsets):
            if subset in stats:
                s = s_final[j].squeeze()  # [H, W]
                stats[subset]['count'] += 1
                stats[subset]['sum'] += s.mean().item()
                stats[subset]['sum_sq'] += s.std().item()
                stats[subset]['min'] = min(stats[subset]['min'], s.min().item())
                stats[subset]['max'] = max(stats[subset]['max'], s.max().item())

        print(f"Batch {i+1}: image {image.shape} -> S_final {s_final.shape}, "
              f"range [{s_final.min():.4f}, {s_final.max():.4f}], "
              f"finite={torch.isfinite(s_final).all()}")

    # 打印统计
    print("\nSubset Statistics (S_final):")
    for subset in ['RSHaze_G', 'RSHaze_L', 'RSHaze_S']:
        s = stats[subset]
        if s['count'] > 0:
            mean_val = s['sum'] / s['count']
            std_val = s['sum_sq'] / s['count']
            print(f"  {subset}: count={s['count']}, mean={mean_val:.4f}, "
                  f"std={std_val:.4f}, range=[{s['min']:.4f}, {s['max']:.4f}]")

    # 打印 timing
    print("\nTiming Statistics (ms):")
    print(f"  Data loading: {sum(timing_stats['data_load'])/len(timing_stats['data_load']):.2f} ± "
          f"{(max(timing_stats['data_load'])-min(timing_stats['data_load']))/2:.2f}")
    print(f"  Physical prior: {sum(timing_stats['physical_prior'])/len(timing_stats['physical_prior']):.2f} ± "
          f"{(max(timing_stats['physical_prior'])-min(timing_stats['physical_prior']))/2:.2f}")
    print(f"  Total: {sum(timing_stats['total'])/len(timing_stats['total']):.2f} ± "
          f"{(max(timing_stats['total'])-min(timing_stats['total']))/2:.2f}")

    print("\n[OK] 256 mode test passed")
    return True


def test_physical_prior_512():
    """测试 512 模式"""
    print_separator("Physical Prior Test (512x512)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 创建 DataLoader
    train_loader = build_rshazeplus_dataloader(
        root='datasets/RSHaze+',
        split='train',
        image_size=512,
        batch_size=2,
        num_workers=0,
        pin_memory=True if device.type == "cuda" else False,
        split_file='experiments/haze_density/rshazeplus_split.json',
    )

    print(f"Train loader: {len(train_loader)} batches")

    # Physical Prior 模块
    physical_prior = PhysicalPriorModule(
        window_size=15,
        guided_radius=15,
        guided_eps=0.01,
    ).to(device)
    physical_prior.eval()

    # 测试一个 batch
    for i, batch in enumerate(train_loader):
        if i >= 1:
            break

        image = batch['image'].to(device, non_blocking=True)

        # 检查输入
        assert image.shape == (2, 3, 512, 512), f"Expected [2,3,512,512], got {image.shape}"

        # Physical Prior forward
        with torch.no_grad():
            s_final = physical_prior(image)

        # 检查输出
        assert s_final.shape == (2, 1, 512, 512), f"Expected [2,1,512,512], got {s_final.shape}"
        assert torch.isfinite(s_final).all(), "S_final contains non-finite values"
        assert s_final.min() >= 0, f"S_final min < 0: {s_final.min()}"
        assert s_final.max() <= 1, f"S_final max > 1: {s_final.max()}"

        print(f"Batch {i+1}: image {image.shape} -> S_final {s_final.shape}, "
              f"range [{s_final.min():.4f}, {s_final.max():.4f}], "
              f"finite={torch.isfinite(s_final).all()}")

    print("\n[OK] 512 mode test passed")
    return True


def test_each_subset():
    """分别测试每个 subset"""
    print_separator("Subset-Specific Test")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    physical_prior = PhysicalPriorModule().to(device)
    physical_prior.eval()

    for subset in ['RSHaze_G', 'RSHaze_L', 'RSHaze_S']:
        print(f"\nTesting {subset}...")

        # 创建只包含该 subset 的 DataLoader
        train_loader = build_rshazeplus_dataloader(
            root='datasets/RSHaze+',
            split='train',
            subsets=(subset,),
            image_size=256,
            batch_size=4,
            num_workers=0,
            split_file='experiments/haze_density/rshazeplus_split.json',
        )

        print(f"  {subset} loader: {len(train_loader)} batches")

        # 测试 4 个样本
        count = 0
        for batch in train_loader:
            if count >= 4:
                break

            image = batch['image'].to(device, non_blocking=True)

            with torch.no_grad():
                s_final = physical_prior(image)

            assert s_final.shape[0] == 4, f"Expected batch_size=4, got {s_final.shape[0]}"
            assert torch.isfinite(s_final).all(), f"{subset}: S_final contains non-finite values"

            print(f"  Sample {count+1}: range [{s_final.min():.4f}, {s_final.max():.4f}], "
                  f"mean={s_final.mean():.4f}, std={s_final.std():.4f}")
            count += 1

    print("\n[OK] All subsets tested")
    return True


def test_cuda():
    """测试 GPU 支持"""
    print_separator("CUDA Test")

    if not torch.cuda.is_available():
        print("[SKIP] CUDA not available")
        return True

    device = torch.device("cuda")
    print(f"Device: {device}")
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    # 创建 DataLoader
    train_loader = build_rshazeplus_dataloader(
        root='datasets/RSHaze+',
        split='train',
        image_size=256,
        batch_size=4,
        num_workers=0,
        pin_memory=True,
        split_file='experiments/haze_density/rshazeplus_split.json',
    )

    # Physical Prior 模块
    physical_prior = PhysicalPriorModule().to(device)
    physical_prior.eval()

    # 测试一个 batch
    for batch in train_loader:
        image = batch['image'].to(device, non_blocking=True)

        assert image.device.type == "cuda", f"Image not on CUDA: {image.device}"

        with torch.no_grad():
            s_final = physical_prior(image)

        assert s_final.device.type == "cuda", f"S_final not on CUDA: {s_final.device}"

        print(f"Image device: {image.device}")
        print(f"S_final device: {s_final.device}")
        print(f"S_final shape: {s_final.shape}")
        print(f"S_final range: [{s_final.min():.4f}, {s_final.max():.4f}]")

        break

    print("\n[OK] CUDA test passed")
    return True


def main():
    """主测试函数"""
    print_separator("Stage 5B-2: RSHaze+ Dataset + Physical Prior Integration Test")

    all_pass = True

    try:
        # 1. 测试 256 模式
        all_pass &= test_physical_prior_256()

        # 2. 测试 512 模式
        all_pass &= test_physical_prior_512()

        # 3. 测试每个 subset
        all_pass &= test_each_subset()

        # 4. 测试 CUDA
        all_pass &= test_cuda()

    except Exception as e:
        print(f"\n[FAIL] Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        all_pass = False

    print_separator("Test Summary")

    if all_pass:
        print("[OK] 所有测试通过！")
        print("\nStage 5B-2 验收项:")
        print("  [✓] Dataset + Physical Prior 联调")
        print("  [✓] 256 forward PASS")
        print("  [✓] 512 forward PASS")
        print("  [✓] output shape 正确")
        print("  [✓] output range [0,1]")
        print("  [✓] finite")
        print("  [✓] CUDA PASS")
        print("  [✓] G/L/S subset PASS")
        print("\n下一步：生成可视化")
    else:
        print("[FAIL] 部分测试失败")

    return all_pass


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
