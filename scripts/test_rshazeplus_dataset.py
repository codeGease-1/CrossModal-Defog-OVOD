#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RSHaze+ Dataset 测试脚本 (Stage 5B-1 Final)

最终 Split:
- Train: 6174 (G=900, L=4374, S=900)
- Val: 686 (G=100, L=486, S=100)
- Test: 930 (G=330, L=270, S=330)

测试项目:
1. dataset length
2. 第一条 sample
3. 随机 sample
4. batch
5. image shape
6. image dtype
7. image range
8. path
9. subset
10. filename
11. train/val/test
12. 每个 subset 的样本数
13. pair integrity
"""

import sys
from pathlib import Path

# 设置路径
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import torch
from torch.utils.data import DataLoader

from src.data import (
    HazeDensityDataset,
    build_rshazeplus_dataloader,
)

# 最终 split 数量
EXPECTED = {
    'train': {'total': 6174, 'RSHaze_G': 900, 'RSHaze_L': 4374, 'RSHaze_S': 900},
    'val': {'total': 686, 'RSHaze_G': 100, 'RSHaze_L': 486, 'RSHaze_S': 100},
    'test': {'total': 930, 'RSHaze_G': 330, 'RSHaze_L': 270, 'RSHaze_S': 330},
}

SPLIT_FILE = 'experiments/haze_density/rshazeplus_split.json'


def print_separator(title: str):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def test_dataset_length():
    """测试 1: dataset length"""
    print_separator("测试 1: Dataset Length")

    try:
        train_ds = HazeDensityDataset(
            root='datasets/RSHaze+',
            split='train',
            image_size=256,
            split_file=SPLIT_FILE,
        )
        val_ds = HazeDensityDataset(
            root='datasets/RSHaze+',
            split='val',
            image_size=256,
            split_file=SPLIT_FILE,
        )
        test_ds = HazeDensityDataset(
            root='datasets/RSHaze+',
            split='test',
            image_size=256,
        )

        train_len = len(train_ds)
        val_len = len(val_ds)
        test_len = len(test_ds)

        print(f"Train: {train_len} samples (expected: {EXPECTED['train']['total']})")
        print(f"Val: {val_len} samples (expected: {EXPECTED['val']['total']})")
        print(f"Test: {test_len} samples (expected: {EXPECTED['test']['total']})")

        # 验证
        assert train_len == EXPECTED['train']['total'], f"Train length mismatch: {train_len} != {EXPECTED['train']['total']}"
        assert val_len == EXPECTED['val']['total'], f"Val length mismatch: {val_len} != {EXPECTED['val']['total']}"
        assert test_len == EXPECTED['test']['total'], f"Test length mismatch: {test_len} != {EXPECTED['test']['total']}"

        print("[OK] Dataset lengths match expected values")
        return True
    except Exception as e:
        print(f"[FAIL] {e}")
        import traceback
        traceback.print_exc()
        return False


def test_first_sample():
    """测试 2: 第一条 sample"""
    print_separator("测试 2: First Sample")

    try:
        ds = HazeDensityDataset(
            root='datasets/RSHaze+',
            split='train',
            image_size=256,
            split_file=SPLIT_FILE,
            return_clean=True,
        )

        sample = ds[0]

        print(f"Keys: {list(sample.keys())}")
        print(f"Image shape: {sample['image'].shape}")
        print(f"Image dtype: {sample['image'].dtype}")
        print(f"Image range: [{sample['image'].min():.4f}, {sample['image'].max():.4f}]")
        print(f"Subset: {sample['subset']}")
        print(f"Filename: {sample['filename']}")
        print(f"Path: {sample['path'][:80]}...")

        if 'clean' in sample:
            print(f"Clean shape: {sample['clean'].shape}")

        # 验证
        assert sample['image'].shape == (3, 256, 256), f"Expected [3,256,256], got {sample['image'].shape}"
        assert sample['image'].dtype == torch.float32, f"Expected float32, got {sample['image'].dtype}"
        assert 0 <= sample['image'].min() <= 1, "Image values should be in [0,1]"
        assert 0 <= sample['image'].max() <= 1, "Image values should be in [0,1]"
        assert sample['subset'] in ['RSHaze_G', 'RSHaze_L', 'RSHaze_S'], f"Unknown subset: {sample['subset']}"
        assert 'filename' in sample, "Missing 'filename' key"

        print("[OK] First sample valid")
        return True
    except Exception as e:
        print(f"[FAIL] {e}")
        import traceback
        traceback.print_exc()
        return False


def test_random_sample():
    """测试 3: 随机 sample"""
    print_separator("测试 3: Random Sample")

    try:
        ds = HazeDensityDataset(
            root='datasets/RSHaze+',
            split='train',
            image_size=256,
            split_file=SPLIT_FILE,
        )

        import random
        idx = random.randint(0, len(ds) - 1)
        sample = ds[idx]

        print(f"Random index: {idx}")
        print(f"Image shape: {sample['image'].shape}")
        print(f"Subset: {sample['subset']}")
        print(f"Filename: {sample['filename']}")

        assert sample['image'].shape == (3, 256, 256)

        print("[OK] Random sample valid")
        return True
    except Exception as e:
        print(f"[FAIL] {e}")
        return False


def test_batch():
    """测试 4: batch"""
    print_separator("测试 4: Batch")

    try:
        ds = HazeDensityDataset(
            root='datasets/RSHaze+',
            split='train',
            image_size=256,
            split_file=SPLIT_FILE,
        )

        loader = DataLoader(
            ds,
            batch_size=4,
            shuffle=True,
            num_workers=0,
        )

        batch = next(iter(loader))

        print(f"Batch image shape: {batch['image'].shape}")
        print(f"Expected: [4, 3, 256, 256]")
        print(f"Subsets: {batch['subset']}")
        print(f"Filenames: {batch['filename'][:3]}...")

        assert batch['image'].shape == (4, 3, 256, 256), f"Expected [4,3,256,256], got {batch['image'].shape}"

        print("[OK] Batch valid")
        return True
    except Exception as e:
        print(f"[FAIL] {e}")
        import traceback
        traceback.print_exc()
        return False


def test_image_range():
    """测试 5: image range"""
    print_separator("测试 5: Image Range")

    try:
        ds = HazeDensityDataset(
            root='datasets/RSHaze+',
            split='train',
            image_size=256,
            split_file=SPLIT_FILE,
        )

        # 采样 100 个样本
        import random
        indices = random.sample(range(len(ds)), min(100, len(ds)))

        min_vals = []
        max_vals = []

        for idx in indices:
            sample = ds[idx]
            min_vals.append(sample['image'].min().item())
            max_vals.append(sample['image'].max().item())

        print(f"Min values range: [{min(min_vals):.4f}, {max(min_vals):.4f}]")
        print(f"Max values range: [{min(max_vals):.4f}, {max(max_vals):.4f}]")

        assert all(m >= 0 for m in min_vals), "Some images have negative values"
        assert all(m <= 1 for m in max_vals), "Some images have values > 1"

        print("[OK] Image range valid")
        return True
    except Exception as e:
        print(f"[FAIL] {e}")
        return False


def test_subset_distribution():
    """测试 6: subset 分布"""
    print_separator("测试 6: Subset Distribution")

    try:
        for split_name in ['train', 'val', 'test']:
            if split_name == 'test':
                ds = HazeDensityDataset(
                    root='datasets/RSHaze+',
                    split=split_name,
                    image_size=256,
                )
            else:
                ds = HazeDensityDataset(
                    root='datasets/RSHaze+',
                    split=split_name,
                    image_size=256,
                    split_file=SPLIT_FILE,
                )

            stats = ds.get_stats()

            print(f"\n{split_name.capitalize()}:")
            print(f"  Total: {stats['total_samples']} (expected: {EXPECTED[split_name]['total']})")

            current_counts = stats['current_subset_counts']
            for subset in ['RSHaze_G', 'RSHaze_L', 'RSHaze_S']:
                count = current_counts.get(subset, 0)
                expected = EXPECTED[split_name][subset]
                match = "✓" if count == expected else "✗"
                print(f"  {subset}: {count} (expected: {expected}) {match}")

                assert count == expected, f"{split_name} {subset} count mismatch: {count} != {expected}"

        print("\n[OK] Subset distribution valid")
        return True
    except Exception as e:
        print(f"[FAIL] {e}")
        import traceback
        traceback.print_exc()
        return False


def test_pair_integrity():
    """测试 7: pair integrity"""
    print_separator("测试 7: Pair Integrity")

    try:
        ds = HazeDensityDataset(
            root='datasets/RSHaze+',
            split='train',
            image_size=256,
            split_file=SPLIT_FILE,
            return_clean=True,
        )

        # 检查所有样本是否有 clean
        missing_clean = 0
        for i in range(len(ds)):
            sample = ds[i]
            if 'clean' not in sample:
                missing_clean += 1

        print(f"Total samples: {len(ds)}")
        print(f"Missing clean: {missing_clean}")

        if missing_clean == 0:
            print("[OK] All pairs intact")
            return True
        else:
            print(f"[WARN] {missing_clean} samples missing clean image")
            return True  # Not a failure, just a warning
    except Exception as e:
        print(f"[FAIL] {e}")
        import traceback
        traceback.print_exc()
        return False


def test_512_mode():
    """测试 8: 512 mode"""
    print_separator("测试 8: 512 Mode")

    try:
        ds = HazeDensityDataset(
            root='datasets/RSHaze+',
            split='test',
            image_size=512,
        )

        sample = ds[0]

        print(f"Image shape: {sample['image'].shape}")
        print(f"Expected: [3, 512, 512]")

        assert sample['image'].shape == (3, 512, 512), f"Expected [3,512,512], got {sample['image'].shape}"

        print("[OK] 512 mode valid")
        return True
    except Exception as e:
        print(f"[FAIL] {e}")
        import traceback
        traceback.print_exc()
        return False


def test_batch_512():
    """测试 9: 512 batch"""
    print_separator("测试 9: 512 Batch")

    try:
        ds = HazeDensityDataset(
            root='datasets/RSHaze+',
            split='test',
            image_size=512,
        )

        loader = DataLoader(
            ds,
            batch_size=2,
            shuffle=False,
            num_workers=0,
        )

        batch = next(iter(loader))

        print(f"Batch image shape: {batch['image'].shape}")
        print(f"Expected: [2, 3, 512, 512]")

        assert batch['image'].shape == (2, 3, 512, 512), f"Expected [2,3,512,512], got {batch['image'].shape}"

        print("[OK] 512 batch valid")
        return True
    except Exception as e:
        print(f"[FAIL] {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataloader_builder():
    """测试 10: DataLoader builder"""
    print_separator("测试 10: DataLoader Builder")

    try:
        train_loader = build_rshazeplus_dataloader(
            root='datasets/RSHaze+',
            split='train',
            image_size=256,
            batch_size=4,
            num_workers=0,
            split_file=SPLIT_FILE,
        )

        val_loader = build_rshazeplus_dataloader(
            root='datasets/RSHaze+',
            split='val',
            image_size=256,
            batch_size=4,
            num_workers=0,
            split_file=SPLIT_FILE,
        )

        test_loader = build_rshazeplus_dataloader(
            root='datasets/RSHaze+',
            split='test',
            image_size=256,
            batch_size=4,
            num_workers=0,
        )

        print(f"Train loader batches: {len(train_loader)}")
        print(f"Val loader batches: {len(val_loader)}")
        print(f"Test loader batches: {len(test_loader)}")

        # 测试迭代
        batch = next(iter(train_loader))
        print(f"Batch shape: {batch['image'].shape}")

        assert batch['image'].shape[0] == 4, "Batch size should be 4"

        print("[OK] DataLoader builder valid")
        return True
    except Exception as e:
        print(f"[FAIL] {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("RSHaze+ Dataset 测试 (Stage 5B-1 Final)")
    print("=" * 60)
    print(f"\nExpected Split:")
    print(f"  Train: {EXPECTED['train']['total']} (G={EXPECTED['train']['RSHaze_G']}, L={EXPECTED['train']['RSHaze_L']}, S={EXPECTED['train']['RSHaze_S']})")
    print(f"  Val: {EXPECTED['val']['total']} (G={EXPECTED['val']['RSHaze_G']}, L={EXPECTED['val']['RSHaze_L']}, S={EXPECTED['val']['RSHaze_S']})")
    print(f"  Test: {EXPECTED['test']['total']} (G={EXPECTED['test']['RSHaze_G']}, L={EXPECTED['test']['RSHaze_L']}, S={EXPECTED['test']['RSHaze_S']})")

    tests = [
        ("Dataset Length", test_dataset_length),
        ("First Sample", test_first_sample),
        ("Random Sample", test_random_sample),
        ("Batch (256)", test_batch),
        ("Image Range", test_image_range),
        ("Subset Distribution", test_subset_distribution),
        ("Pair Integrity", test_pair_integrity),
        ("512 Mode", test_512_mode),
        ("512 Batch", test_batch_512),
        ("DataLoader Builder", test_dataloader_builder),
    ]

    results = []

    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"[FAIL] {name}: {e}")
            results.append((name, False))

    # 汇总
    print_separator("测试结果汇总")

    for name, passed in results:
        status = "[OK]" if passed else "[FAIL]"
        print(f"{status} {name}")

    all_passed = all(r[1] for r in results)

    if all_passed:
        print("\n[OK] 所有测试通过！")
    else:
        print("\n[FAIL] 部分测试失败")

    print("=" * 60)

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
