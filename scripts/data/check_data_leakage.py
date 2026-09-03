#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据泄漏检查脚本

检查 train/val/test split 之间是否存在重复的原始图像。

使用方法:
    !python scripts/check_data_leakage.py
"""

import sys
from pathlib import Path

# 设置路径
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data import HazeDensityDataset


def print_separator(title: str):
    """打印分隔线"""
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def get_image_identifiers(dataset):
    """
    获取数据集的所有图像标识符

    使用 (雾级别目录，文件名) 作为唯一标识
    """
    identifiers = set()

    for i in range(len(dataset)):
        sample = dataset[i]
        if 'path' in sample:
            path = Path(sample['path'])
            # 使用父目录名 (如 RSHaze_G/train/synhazypng) + 文件名
            parent_parts = path.parent.parts
            # 获取最后三个部分：RSHaze_G, train, synhazypng
            if len(parent_parts) >= 3:
                fog_level = parent_parts[-3] if len(parent_parts) >= 3 else 'unknown'
                split_name = parent_parts[-2] if len(parent_parts) >= 2 else 'unknown'
                identifier = (fog_level, split_name, path.name)
            else:
                identifier = (str(path.parent), path.name)
            identifiers.add(identifier)

    return identifiers


def check_leakage():
    """检查数据泄漏"""
    print_separator("数据泄漏检查")

    try:
        # 加载数据集
        print("\n加载数据集...")

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

        print(f"  Train dataset: {len(train_ds)} images")
        print(f"  Test dataset: {len(test_ds)} images")

        # 获取标识符
        print("\n提取图像标识符...")
        train_ids = get_image_identifiers(train_ds)
        test_ids = get_image_identifiers(test_ds)

        print(f"  Train unique identifiers: {len(train_ids)}")
        print(f"  Test unique identifiers: {len(test_ids)}")

        # 检查重叠
        print("\n检查重叠...")
        overlap = train_ids & test_ids

        if len(overlap) == 0:
            print("\n[OK] 未发现数据泄漏!")
            print("    Train 和 Test 集合完全独立。")
            return True
        else:
            print(f"\n[FAIL] 发现 {len(overlap)} 个重复图像!")
            print("\n重复图像列表:")
            for i, identifier in enumerate(list(overlap)[:20]):
                print(f"    {i+1}. {identifier}")
            if len(overlap) > 20:
                print(f"    ... 还有 {len(overlap) - 20} 个重复图像")
            return False

    except Exception as e:
        print(f"\n[FAIL] 错误：{e}")
        import traceback
        traceback.print_exc()
        return False


def check_internal_duplicates():
    """检查数据集内部的重复"""
    print_separator("内部重复检查")

    try:
        # 检查 train 集内部
        train_ds = HazeDensityDataset(
            root='datasets/RSHaze+',
            split='train',
            image_size=256,
            return_path=True,
        )

        train_ids = get_image_identifiers(train_ds)

        print(f"\nTrain 集:")
        print(f"  总样本数：{len(train_ds)}")
        print(f"  唯一标识符：{len(train_ids)}")

        if len(train_ds) != len(train_ids):
            print(f"  [WARN] 发现 {len(train_ds) - len(train_ids)} 个内部重复")
        else:
            print(f"  [OK] 无内部重复")

        # 检查 test 集内部
        test_ds = HazeDensityDataset(
            root='datasets/RSHaze+',
            split='test',
            image_size=256,
            return_path=True,
        )

        test_ids = get_image_identifiers(test_ds)

        print(f"\nTest 集:")
        print(f"  总样本数：{len(test_ds)}")
        print(f"  唯一标识符：{len(test_ids)}")

        if len(test_ds) != len(test_ids):
            print(f"  [WARN] 发现 {len(test_ds) - len(test_ids)} 个内部重复")
        else:
            print(f"  [OK] 无内部重复")

        # 总结
        no_duplicates = (len(train_ds) == len(train_ids)) and (len(test_ds) == len(test_ids))

        if no_duplicates:
            print("\n[OK] 所有数据集内部无重复!")
        else:
            print("\n[WARN] 部分数据集存在内部重复")

        return no_duplicates

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

    results.append(("Train/Test 泄漏", check_leakage()))
    results.append(("内部重复", check_internal_duplicates()))

    # 汇总
    print_separator("检查结果汇总")

    for name, passed in results:
        status = "[OK]" if passed else "[FAIL]"
        print(f"{status} {name}")

    all_passed = all(r[1] for r in results)

    if all_passed:
        print("\n[OK] 所有检查通过！数据无泄漏。")
    else:
        print("\n[FAIL] 发现数据问题，请检查")

    print("=" * 60)

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
