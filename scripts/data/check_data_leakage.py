#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据泄漏检查脚本

检查 train/val/test 是否复用了同一原始路径，以及各集合内部重复。
路径不同不能证明内容或场景独立；内容/场景泄漏需另行核查。

使用方法:
    python scripts/data/check_data_leakage.py
"""

import sys
from pathlib import Path

# 设置路径
project_root = Path(__file__).parent.parent.parent  # scripts/data/ -> 项目根
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data import HazeDensityDataset


def print_separator(title: str):
    """打印分隔线"""
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def get_image_identifiers(dataset):
    """只读样本元数据，不解码图片或触发随机增强。"""
    return {str(Path(item['hazy_path']).resolve()) for item in dataset.image_list}


def check_leakage(root='datasets/RSHaze+', split_file=None):
    """检查集合之间的原始路径复用。"""
    print_separator("原始路径重叠检查")

    try:
        # 加载数据集
        print("\n加载数据集...")

        datasets = {
            split: HazeDensityDataset(root=root, split=split, split_file=split_file)
            for split in ('train', 'val', 'test')
        }
        identifiers = {split: get_image_identifiers(ds) for split, ds in datasets.items()}
        success = True
        for left, right in (('train', 'val'), ('train', 'test'), ('val', 'test')):
            overlap = identifiers[left] & identifiers[right]
            print(f"  {left}/{right}: {len(overlap)} shared source paths")
            success = success and not overlap
        print("仅检查原始路径；内容重复和同场景泄漏尚未核查。")
        return success

    except Exception as e:
        print(f"\n[FAIL] 错误：{e}")
        import traceback
        traceback.print_exc()
        return False


def check_internal_duplicates(root='datasets/RSHaze+', split_file=None):
    """检查数据集内部的重复"""
    print_separator("内部重复检查")

    try:
        success = True
        for split in ('train', 'val', 'test'):
            dataset = HazeDensityDataset(root=root, split=split, split_file=split_file)
            duplicate_count = len(dataset) - len(get_image_identifiers(dataset))
            print(f"  {split}: {duplicate_count} duplicate source paths")
            success = success and duplicate_count == 0
        return success

    except Exception as e:
        print(f"\n[FAIL] 错误：{e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("数据泄漏检查")
    print("=" * 60)

    # 运行检查
    results = []

    results.append(("Train/Val/Test 原始路径重叠", check_leakage()))
    results.append(("内部重复", check_internal_duplicates()))

    # 汇总
    print_separator("检查结果汇总")

    for name, passed in results:
        status = "[OK]" if passed else "[FAIL]"
        print(f"{status} {name}")

    all_passed = all(r[1] for r in results)

    if all_passed:
        print("\n[OK] 原始路径检查通过；内容/场景是否泄漏尚待核查。")
    else:
        print("\n[FAIL] 发现数据问题，请检查")

    print("=" * 60)

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
