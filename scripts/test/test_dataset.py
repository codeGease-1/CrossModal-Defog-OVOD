#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据集测试脚本

【在 Colab 或本地执行】

测试内容:
1. Dataset length
2. 单样本 shape
3. batch shape
4. dtype
5. image range [0, 1]
6. RGB channel
7. train/test split
8. 随机 crop 后尺寸
9. path 是否正确
10. DataLoader 迭代

使用方法:
    !python scripts/test_dataset.py
"""

import sys
from pathlib import Path

# 设置路径
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import torch
from torch.utils.data import DataLoader
from src.data import HazeDensityDataset


def print_separator(title: str):
    """打印分隔线"""
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def test_dataset_length():
    """1. Dataset length"""
    print_separator("1. Dataset Length Test")

    try:
        train_ds = HazeDensityDataset(
            root='datasets/RSHaze+',
            split='train',
            image_size=256,
            return_path=False,
        )
        test_ds = HazeDensityDataset(
            root='datasets/RSHaze+',
            split='test',
            image_size=256,
            return_path=False,
        )

        print(f"  Train dataset length: {len(train_ds)}")
        print(f"  Test dataset length: {len(test_ds)}")

        if len(train_ds) > 0 and len(test_ds) > 0:
            print("[OK] Dataset length test passed!")
            return True
        else:
            print("[FAIL] Dataset length is 0!")
            return False
    except Exception as e:
        print(f"[FAIL] Error: {e}")
        return False


def test_single_sample():
    """2. Single sample shape"""
    print_separator("2. Single Sample Shape Test")

    try:
        ds = HazeDensityDataset(
            root='datasets/RSHaze+',
            split='train',
            image_size=256,
            return_path=True,
        )

        sample = ds[0]

        image = sample['image']
        print(f"  Image shape: {image.shape}")
        print(f"  Image dtype: {image.dtype}")
        print(f"  Image range: [{image.min():.4f}, {image.max():.4f}]")
        print(f"  Has path: {'path' in sample}")
        print(f"  Fog level: {sample.get('fog_level', 'N/A')}")

        # 检查
        passed = (
            image.shape == torch.Size([3, 256, 256]) and
            image.dtype == torch.float32 and
            image.min() >= 0 and image.max() <= 1
        )

        if passed:
            print("[OK] Single sample test passed!")
        else:
            print("[FAIL] Single sample test failed!")

        return passed
    except Exception as e:
        print(f"[FAIL] Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_batch_shape():
    """3. Batch shape"""
    print_separator("3. Batch Shape Test")

    try:
        ds = HazeDensityDataset(
            root='datasets/RSHaze+',
            split='train',
            image_size=256,
            return_path=False,
        )

        loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)

        for batch in loader:
            image = batch['image']
            print(f"  Batch image shape: {image.shape}")
            print(f"  Batch fog_levels: {batch.get('fog_level', ['N/A'])}")

            passed = image.shape == torch.Size([4, 3, 256, 256])

            if passed:
                print("[OK] Batch shape test passed!")
            else:
                print("[FAIL] Batch shape test failed!")

            return passed
    except Exception as e:
        print(f"[FAIL] Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_image_range():
    """4. Image range [0, 1]"""
    print_separator("4. Image Range Test")

    try:
        ds = HazeDensityDataset(
            root='datasets/RSHaze+',
            split='train',
            image_size=256,
            return_path=False,
        )

        loader = DataLoader(ds, batch_size=8, shuffle=True, num_workers=0)

        all_passed = True
        for i, batch in enumerate(loader):
            image = batch['image']
            min_val = image.min().item()
            max_val = image.max().item()

            if min_val < 0 or max_val > 1:
                print(f"  [FAIL] Batch {i}: range [{min_val:.4f}, {max_val:.4f}]")
                all_passed = False

        if all_passed:
            print(f"  All batches in range [0, 1]")
            print("[OK] Image range test passed!")
        else:
            print("[FAIL] Image range test failed!")

        return all_passed
    except Exception as e:
        print(f"[FAIL] Error: {e}")
        return False


def test_rgb_channel():
    """5. RGB channel"""
    print_separator("5. RGB Channel Test")

    try:
        ds = HazeDensityDataset(
            root='datasets/RSHaze+',
            split='train',
            image_size=256,
            return_path=False,
        )

        sample = ds[0]
        image = sample['image']

        channels = image.shape[0]
        print(f"  Number of channels: {channels}")

        if channels == 3:
            print("[OK] RGB channel test passed!")
            return True
        else:
            print(f"[FAIL] Expected 3 channels, got {channels}")
            return False
    except Exception as e:
        print(f"[FAIL] Error: {e}")
        return False


def test_train_test_split():
    """6. Train/Test split"""
    print_separator("6. Train/Test Split Test")

    try:
        train_ds = HazeDensityDataset(
            root='datasets/RSHaze+',
            split='train',
            image_size=256,
            return_path=True,
        )
        test_ds = HazeDensityDataset(
            root='datasets/RSHaze+',
            split='test',
            image_size=256,
            return_path=True,
        )

        # 获取所有路径
        train_paths = set()
        for i in range(len(train_ds)):
            path = Path(train_ds[i]['path'])
            # 使用父目录 + 文件名作为唯一标识
            train_paths.add((path.parent.name, path.name))

        test_paths = set()
        for i in range(len(test_ds)):
            path = Path(test_ds[i]['path'])
            test_paths.add((path.parent.name, path.name))

        # 检查重叠
        overlap = train_paths & test_paths

        print(f"  Train images: {len(train_paths)}")
        print(f"  Test images: {len(test_paths)}")
        print(f"  Overlap: {len(overlap)}")

        if len(overlap) == 0:
            print("[OK] Train/Test split test passed!")
            return True
        else:
            print(f"[FAIL] Found {len(overlap)} overlapping images!")
            for p in list(overlap)[:5]:
                print(f"    {p}")
            return False
    except Exception as e:
        print(f"[FAIL] Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_random_crop():
    """7. Random crop size"""
    print_separator("7. Random Crop Size Test")

    try:
        # 训练集使用随机 crop
        ds = HazeDensityDataset(
            root='datasets/RSHaze+',
            split='train',
            image_size=256,
            return_path=False,
        )

        # 多次采样检查尺寸一致性
        sizes = set()
        for i in range(10):
            sample = ds[i]
            sizes.add(tuple(sample['image'].shape))

        print(f"  Sample shapes (10 samples): {sizes}")

        if len(sizes) == 1 and list(sizes)[0] == (3, 256, 256):
            print("[OK] Random crop size test passed!")
            return True
        else:
            print("[FAIL] Inconsistent crop sizes!")
            return False
    except Exception as e:
        print(f"[FAIL] Error: {e}")
        return False


def test_dataloader_iteration():
    """8. DataLoader iteration"""
    print_separator("8. DataLoader Iteration Test")

    try:
        ds = HazeDensityDataset(
            root='datasets/RSHaze+',
            split='train',
            image_size=256,
            return_path=False,
        )

        loader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=0)

        batch_count = 0
        for batch in loader:
            batch_count += 1
            if batch_count <= 2:
                print(f"  Batch {batch_count}: {batch['image'].shape}")

        expected_batches = (len(ds) + 3) // 4
        print(f"  Total batches: {batch_count} (expected ~{expected_batches})")

        if batch_count > 0:
            print("[OK] DataLoader iteration test passed!")
            return True
        else:
            print("[FAIL] No batches produced!")
            return False
    except Exception as e:
        print(f"[FAIL] Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_fog_levels():
    """9. Fog levels distribution"""
    print_separator("9. Fog Levels Distribution Test")

    try:
        train_ds = HazeDensityDataset(
            root='datasets/RSHaze+',
            split='train',
            image_size=256,
            return_path=False,
        )

        stats = train_ds.get_stats()
        print(f"  Dataset: {stats['dataset']}")
        print(f"  Split: {stats['split']}")
        print(f"  Total images: {stats['total_images']}")
        print(f"  Fog counts: {stats['fog_counts']}")

        if stats['total_images'] > 0 and len(stats['fog_counts']) > 0:
            print("[OK] Fog levels test passed!")
            return True
        else:
            print("[FAIL] No fog level data!")
            return False
    except Exception as e:
        print(f"[FAIL] Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("雾密度数据集测试")
    print("=" * 60)

    # 运行测试
    results = []

    results.append(("Dataset Length", test_dataset_length()))
    results.append(("Single Sample", test_single_sample()))
    results.append(("Batch Shape", test_batch_shape()))
    results.append(("Image Range", test_image_range()))
    results.append(("RGB Channel", test_rgb_channel()))
    results.append(("Train/Test Split", test_train_test_split()))
    results.append(("Random Crop", test_random_crop()))
    results.append(("DataLoader Iteration", test_dataloader_iteration()))
    results.append(("Fog Levels", test_fog_levels()))

    # 汇总
    print_separator("测试结果汇总")

    for name, passed in results:
        status = "[OK]" if passed else "[FAIL]"
        print(f"{status} {name}")

    all_passed = all(r[1] for r in results)

    if all_passed:
        print("\n[OK] 所有测试通过！")
    else:
        print("\n[FAIL] 部分测试未通过，请检查")

    print("=" * 60)

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
