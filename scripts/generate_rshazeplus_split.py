#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RSHaze+ Split 生成脚本 (修复版)

生成 train/val/test split 并保存到 JSON 文件。

规则:
1. test: 使用官方 test
2. train/val: 从官方 train 中按 90/10 划分 (seed=42)
3. 按 subset 分别划分，保持分布
4. 唯一键使用 (subset, filename)

JSON Schema:
{
  "train": [
    {"subset": "RSHaze_G", "filename": "1.png"}
  ],
  "val": [...],
  "test": [...],
  "metadata": {...}
}
"""

import sys
from pathlib import Path
import json
import random


def print_separator(title: str):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def scan_rshazeplus_dataset(root: str):
    """
    直接扫描 RSHaze+ 数据集，不依赖 Dataset 类

    Returns:
        dict: {
            'RSHaze_G': {'train': ['1.png', ...], 'test': [...]},
            'RSHaze_L': {...},
            'RSHaze_S': {...}
        }
    """
    root_path = Path(root)
    subsets = ['RSHaze_G', 'RSHaze_L', 'RSHaze_S']
    supported_formats = {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}

    manifest = {}

    for subset in subsets:
        subset_dir = root_path / subset
        if not subset_dir.exists():
            print(f"[WARN] Subset directory not found: {subset_dir}")
            manifest[subset] = {'train': [], 'test': []}
            continue

        manifest[subset] = {'train': [], 'test': []}

        for split_name in ['train', 'test']:
            split_dir = subset_dir / split_name
            hazy_dir = split_dir / 'synhazypng'

            if not hazy_dir.exists():
                print(f"[WARN] Directory not found: {hazy_dir}")
                continue

            # 获取所有 hazy 文件名
            files = sorted([
                f.name for f in hazy_dir.iterdir()
                if f.suffix.lower() in supported_formats and f.is_file()
            ])

            manifest[subset][split_name] = files
            print(f"  {subset}/{split_name}: {len(files)} files")

    return manifest


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

    # 扫描数据集
    print("\n扫描数据集...")
    manifest = scan_rshazeplus_dataset(root)

    # 统计总数
    total_train = sum(len(m['train']) for m in manifest.values())
    total_test = sum(len(m['test']) for m in manifest.values())

    print(f"\n官方 train 总计：{total_train} samples")
    print(f"官方 test 总计：{total_test} samples")

    # 按 subset 划分 train/val
    subsets = ['RSHaze_G', 'RSHaze_L', 'RSHaze_S']

    train_list = []  # [{"subset": "...", "filename": "..."}]
    val_list = []
    test_list = []

    # 统计各 subset 数量
    subset_train_counts = {}
    subset_val_counts = {}
    subset_test_counts = {}

    for subset in subsets:
        subset_data = manifest[subset]
        train_files = subset_data['train']
        test_files = subset_data['test']

        print(f"\n{subset}:")
        print(f"  官方 train: {len(train_files)}")
        print(f"  官方 test: {len(test_files)}")

        # 打乱 train files
        random.shuffle(train_files)

        # 划分 train/val
        n_val = max(1, int(len(train_files) * val_ratio))
        n_train = len(train_files) - n_val

        val_files = train_files[:n_val]
        train_files_only = train_files[n_val:]

        # 添加到列表
        for filename in train_files_only:
            train_list.append({'subset': subset, 'filename': filename})

        for filename in val_files:
            val_list.append({'subset': subset, 'filename': filename})

        for filename in test_files:
            test_list.append({'subset': subset, 'filename': filename})

        # 统计
        subset_train_counts[subset] = n_train
        subset_val_counts[subset] = n_val
        subset_test_counts[subset] = len(test_files)

        print(f"  Train: {n_train}, Val: {n_val}")

    # 统计
    print_separator("Split 统计")

    print(f"\nTrain: {len(train_list)} samples")
    print(f"Val: {len(val_list)} samples")
    print(f"Test: {len(test_list)} samples")

    # 各 subset 分布
    print(f"\nTrain subset 分布:")
    for subset in subsets:
        print(f"  {subset}: {subset_train_counts[subset]}")

    print(f"\nVal subset 分布:")
    for subset in subsets:
        print(f"  {subset}: {subset_val_counts[subset]}")

    print(f"\nTest subset 分布:")
    for subset in subsets:
        print(f"  {subset}: {subset_test_counts[subset]}")

    # 保存
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    split_data = {
        'train': train_list,
        'val': val_list,
        'test': test_list,
        'metadata': {
            'val_ratio': val_ratio,
            'seed': seed,
            'total': {
                'train': len(train_list),
                'val': len(val_list),
                'test': len(test_list),
            },
            'subset_counts': {
                'train': subset_train_counts,
                'val': subset_val_counts,
                'test': subset_test_counts,
            },
        }
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(split_data, f, indent=2, ensure_ascii=False)

    print(f"\n保存 split 到：{output_path}")

    # 验证无重叠 - 使用 (subset, filename) 作为唯一键
    print_separator("重叠检查")

    train_keys = set((item['subset'], item['filename']) for item in train_list)
    val_keys = set((item['subset'], item['filename']) for item in val_list)
    test_keys = set((item['subset'], item['filename']) for item in test_list)

    train_val_overlap = train_keys & val_keys
    train_test_overlap = train_keys & test_keys
    val_test_overlap = val_keys & test_keys

    all_ok = True

    if len(train_val_overlap) == 0:
        print("[OK] Train/Val: No overlap")
    else:
        print(f"[FAIL] Train/Val overlap: {len(train_val_overlap)}")
        all_ok = False

    if len(train_test_overlap) == 0:
        print("[OK] Train/Test: No overlap")
    else:
        print(f"[FAIL] Train/Test overlap: {len(train_test_overlap)}")
        all_ok = False

    if len(val_test_overlap) == 0:
        print("[OK] Val/Test: No overlap")
    else:
        print(f"[FAIL] Val/Test overlap: {len(val_test_overlap)}")
        all_ok = False

    # 检查内部重复
    print("\n内部重复检查:")
    if len(train_list) == len(train_keys):
        print("[OK] Train: No internal duplicates")
    else:
        print(f"[FAIL] Train internal duplicates: {len(train_list) - len(train_keys)}")
        all_ok = False

    if len(val_list) == len(val_keys):
        print("[OK] Val: No internal duplicates")
    else:
        print(f"[FAIL] Val internal duplicates: {len(val_list) - len(val_keys)}")
        all_ok = False

    if len(test_list) == len(test_keys):
        print("[OK] Test: No internal duplicates")
    else:
        print(f"[FAIL] Test internal duplicates: {len(test_list) - len(test_keys)}")
        all_ok = False

    print_separator("Split 生成完成")

    return split_data, all_ok


def main():
    """主函数"""
    split_data, success = generate_split(
        root='datasets/RSHaze+',
        val_ratio=0.1,
        seed=42,
        output_file='experiments/haze_density/rshazeplus_split.json',
    )

    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
