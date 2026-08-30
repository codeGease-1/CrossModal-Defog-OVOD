#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RSHaze+ Split 验证脚本 (Stage 5B-1)

检查:
1. train ∩ val = ∅
2. train ∩ test = ∅
3. val ∩ test = ∅

基于原始图像 ID 检查，不是 patch filename。
"""

import sys
from pathlib import Path

# 设置路径
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data import HazeDensityDataset


def print_separator(title: str):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def get_all_ids(dataset):
    """获取数据集所有 ID"""
    return set(dataset.get_all_ids())


def check_split_integrity():
    """检查 split 完整性"""
    print_separator("Split 完整性检查")

    try:
        # 加载数据集
        print("\n加载数据集...")

        train_ds = HazeDensityDataset(
            root='datasets/RSHaze+',
            split='train',
            image_size=256,
            split_file='experiments/haze_density/rshazeplus_split.json',
        )

        val_ds = HazeDensityDataset(
            root='datasets/RSHaze+',
            split='val',
            image_size=256,
            split_file='experiments/haze_density/rshazeplus_split.json',
        )

        test_ds = HazeDensityDataset(
            root='datasets/RSHaze+',
            split='test',
            image_size=256,
        )

        print(f"Train: {len(train_ds)} samples")
        print(f"Val: {len(val_ds)} samples")
        print(f"Test: {len(test_ds)} samples")

        # 获取 ID 集合
        print("\n提取 ID 集合...")
        train_ids = get_all_ids(train_ds)
        val_ids = get_all_ids(val_ds)
        test_ids = get_all_ids(test_ds)

        print(f"Train unique IDs: {len(train_ids)}")
        print(f"Val unique IDs: {len(val_ids)}")
        print(f"Test unique IDs: {len(test_ids)}")

        # 检查重叠
        errors = []

        # train ∩ val
        train_val_overlap = train_ids & val_ids
        if len(train_val_overlap) > 0:
            errors.append(f"Train/Val overlap: {len(train_val_overlap)} samples")
            print(f"\n[FAIL] Train/Val overlap: {len(train_val_overlap)} samples")
            for id_ in list(train_val_overlap)[:5]:
                print(f"  - {id_}")
        else:
            print("\n[OK] Train/Val: No overlap")

        # train ∩ test
        train_test_overlap = train_ids & test_ids
        if len(train_test_overlap) > 0:
            errors.append(f"Train/Test overlap: {len(train_test_overlap)} samples")
            print(f"[FAIL] Train/Test overlap: {len(train_test_overlap)} samples")
            for id_ in list(train_test_overlap)[:5]:
                print(f"  - {id_}")
        else:
            print("[OK] Train/Test: No overlap")

        # val ∩ test
        val_test_overlap = val_ids & test_ids
        if len(val_test_overlap) > 0:
            errors.append(f"Val/Test overlap: {len(val_test_overlap)} samples")
            print(f"[FAIL] Val/Test overlap: {len(val_test_overlap)} samples")
            for id_ in list(val_test_overlap)[:5]:
                print(f"  - {id_}")
        else:
            print("[OK] Val/Test: No overlap")

        # 检查内部重复
        print("\n检查内部重复...")
        if len(train_ds) != len(train_ids):
            errors.append(f"Train internal duplicates: {len(train_ds) - len(train_ids)}")
            print(f"[WARN] Train internal duplicates: {len(train_ds) - len(train_ids)}")
        else:
            print("[OK] Train: No internal duplicates")

        if len(val_ds) != len(val_ids):
            errors.append(f"Val internal duplicates: {len(val_ds) - len(val_ids)}")
            print(f"[WARN] Val internal duplicates: {len(val_ds) - len(val_ids)}")
        else:
            print("[OK] Val: No internal duplicates")

        if len(test_ds) != len(test_ids):
            errors.append(f"Test internal duplicates: {len(test_ds) - len(test_ids)}")
            print(f"[WARN] Test internal duplicates: {len(test_ds) - len(test_ids)}")
        else:
            print("[OK] Test: No internal duplicates")

        # 汇总
        if len(errors) == 0:
            print("\n[OK] Split 完整性检查通过！")
            return True
        else:
            print(f"\n[FAIL] 发现 {len(errors)} 个问题:")
            for error in errors:
                print(f"  - {error}")
            return False

    except Exception as e:
        print(f"\n[FAIL] 错误：{e}")
        import traceback
        traceback.print_exc()
        return False


def check_subset_distribution():
    """检查各 subset 分布"""
    print_separator("Subset 分布检查")

    try:
        train_ds = HazeDensityDataset(
            root='datasets/RSHaze+',
            split='train',
            image_size=256,
            split_file='experiments/haze_density/rshazeplus_split.json',
        )

        val_ds = HazeDensityDataset(
            root='datasets/RSHaze+',
            split='val',
            image_size=256,
            split_file='experiments/haze_density/rshazeplus_split.json',
        )

        test_ds = HazeDensityDataset(
            root='datasets/RSHaze+',
            split='test',
            image_size=256,
        )

        # 统计各 split 的 subset 分布
        for name, ds in [('Train', train_ds), ('Val', val_ds), ('Test', test_ds)]:
            subset_counts = {}
            for i in range(len(ds)):
                sample = ds[i]
                subset = sample['subset']
                subset_counts[subset] = subset_counts.get(subset, 0) + 1

            print(f"\n{name}:")
            for subset in ['RSHaze_G', 'RSHaze_L', 'RSHaze_S']:
                count = subset_counts.get(subset, 0)
                print(f"  {subset}: {count}")

        return True

    except Exception as e:
        print(f"[FAIL] {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("RSHaze+ Split 验证 (Stage 5B-1)")
    print("=" * 60)

    results = []

    results.append(("Split 完整性", check_split_integrity()))
    results.append(("Subset 分布", check_subset_distribution()))

    # 汇总
    print_separator("验证结果汇总")

    for name, passed in results:
        status = "[OK]" if passed else "[FAIL]"
        print(f"{status} {name}")

    all_passed = all(r[1] for r in results)

    if all_passed:
        print("\n[OK] 所有验证通过！")
    else:
        print("\n[FAIL] 部分验证失败")

    print("=" * 60)

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
