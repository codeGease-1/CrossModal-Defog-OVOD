#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RSHaze+ Split 生成脚本 (Stage 5B-1)

生成 train/val/test split 并保存到 JSON 文件。

规则:
1. test: 使用官方 test
2. train/val: 从官方 train 中按 90/10 划分 (seed=42)
3. 按 subset 分别划分，保持分布
"""

import sys
from pathlib import Path
import json
import random

# 设置路径
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data import HazeDensityDataset


def print_separator(title: str):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def generate_split(
    root: str = 'datasets/RSHaze+',
    val_ratio: float = 0.1,
    seed: int = 42,
    output_file: str = 'experiments/haze_density/rshazeplus_split.json',
):
    """
    生成 split

    Args:
        root: 数据集根目录
        val_ratio: val 比例
        seed: 随机种子
        output_file: 输出文件路径
    """
    print_separator("RSHaze+ Split 生成")

    # 固定随机种子
    random.seed(seed)

    # 加载所有官方 train 数据
    print("\n加载官方 train 数据...")

    all_ds = HazeDensityDataset(
        root=root,
        split='train',  # 这里实际上是加载所有官方 train
        image_size=256,
    )

    # 访问内部数据获取所有 pairs
    train_items = [
        item for item in all_ds._all_pairs
        if item['official_split'] == 'train'
    ]

    test_items = [
        item for item in all_ds._all_pairs
        if item['official_split'] == 'test'
    ]

    print(f"官方 train: {len(train_items)} samples")
    print(f"官方 test: {len(test_items)} samples")

    # 按 subset 划分 train/val
    subsets = ['RSHaze_G', 'RSHaze_L', 'RSHaze_S']

    train_ids = []
    val_ids = []

    for subset in subsets:
        subset_items = [
            item for item in train_items
            if item['subset'] == subset
        ]

        print(f"\n{subset}: {len(subset_items)} samples")

        # 打乱
        random.shuffle(subset_items)

        # 划分
        n_val = max(1, int(len(subset_items) * val_ratio))
        n_train = len(subset_items) - n_val

        val_subset = subset_items[:n_val]
        train_subset = subset_items[n_val:]

        for item in train_subset:
            train_ids.append(item['id'])

        for item in val_subset:
            val_ids.append(item['id'])

        print(f"  Train: {n_train}, Val: {n_val}")

    # test IDs
    test_ids = [item['id'] for item in test_items]

    # 统计
    print_separator("Split 统计")

    print(f"\nTrain: {len(train_ids)} samples")
    print(f"Val: {len(val_ids)} samples")
    print(f"Test: {len(test_ids)} samples")

    # 各 subset 分布
    for split_name, ids in [('Train', train_ids), ('Val', val_ids), ('Test', test_ids)]:
        subset_counts = {}
        for id_ in ids:
            subset = id_.split('_')[0]
            subset_counts[subset] = subset_counts.get(subset, 0) + 1
        print(f"\n{split_name} subset 分布:")
        for subset in subsets:
            print(f"  {subset}: {subset_counts.get(subset, 0)}")

    # 保存
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    split_data = {
        'train': sorted(train_ids),
        'val': sorted(val_ids),
        'test': sorted(test_ids),
        'val_ratio': val_ratio,
        'seed': seed,
        'total': {
            'train': len(train_ids),
            'val': len(val_ids),
            'test': len(test_ids),
        },
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(split_data, f, indent=2, ensure_ascii=False)

    print(f"\n保存 split 到：{output_path}")

    # 验证无重叠
    print_separator("重叠检查")

    train_set = set(train_ids)
    val_set = set(val_ids)
    test_set = set(test_ids)

    train_val_overlap = train_set & val_set
    train_test_overlap = train_set & test_set
    val_test_overlap = val_set & test_set

    if len(train_val_overlap) == 0:
        print("[OK] Train/Val: No overlap")
    else:
        print(f"[FAIL] Train/Val overlap: {len(train_val_overlap)}")

    if len(train_test_overlap) == 0:
        print("[OK] Train/Test: No overlap")
    else:
        print(f"[FAIL] Train/Test overlap: {len(train_test_overlap)}")

    if len(val_test_overlap) == 0:
        print("[OK] Val/Test: No overlap")
    else:
        print(f"[FAIL] Val/Test overlap: {len(val_test_overlap)}")

    print_separator("Split 生成完成")

    return split_data


def main():
    """主函数"""
    generate_split(
        root='datasets/RSHaze+',
        val_ratio=0.1,
        seed=42,
        output_file='experiments/haze_density/rshazeplus_split.json',
    )

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
