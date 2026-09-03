#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RSHaze+ Split 验证脚本 (修复版)

检查:
1. train ∩ val = ∅
2. train ∩ test = ∅
3. val ∩ test = ∅

基于 (subset, filename) 唯一键检查。
"""

import sys
from pathlib import Path
import json


def print_separator(title: str):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def verify_split_json(split_file: str = 'experiments/haze_density/rshazeplus_split.json'):
    """直接验证 JSON split 文件"""
    print_separator("Split JSON 验证")

    split_path = Path(split_file)
    if not split_path.exists():
        print(f"[FAIL] Split file not found: {split_file}")
        print("请先运行 generate_rshazeplus_split.py")
        return False

    try:
        with open(split_path, 'r', encoding='utf-8') as f:
            split_data = json.load(f)

        # 检查 schema
        print("\n检查 JSON schema...")
        required_keys = {'train', 'val', 'test'}
        if not required_keys.issubset(split_data.keys()):
            print(f"[FAIL] Missing keys: {required_keys - set(split_data.keys())}")
            return False

        # 检查格式
        train_list = split_data['train']
        val_list = split_data['val']
        test_list = split_data['test']

        print(f"Train entries: {len(train_list)}")
        print(f"Val entries: {len(val_list)}")
        print(f"Test entries: {len(test_list)}")

        # 检查是否为新格式
        if len(train_list) > 0 and isinstance(train_list[0], dict):
            print("\n[OK] 使用新格式：[{subset, filename}]")
            required_item_keys = {'subset', 'filename'}
            if not required_item_keys.issubset(train_list[0].keys()):
                print(f"[FAIL] Missing item keys: {required_item_keys - set(train_list[0].keys())}")
                return False
        else:
            print("\n[WARN] 使用旧格式：[id1, id2, ...]")

        # 提取唯一键 (subset, filename)
        def extract_keys(item_list):
            keys = set()
            for item in item_list:
                if isinstance(item, dict):
                    keys.add((item['subset'], item['filename']))
                else:
                    # 旧格式
                    keys.add(item)
            return keys

        train_keys = extract_keys(train_list)
        val_keys = extract_keys(val_list)
        test_keys = extract_keys(test_list)

        print(f"\nTrain unique keys: {len(train_keys)}")
        print(f"Val unique keys: {len(val_keys)}")
        print(f"Test unique keys: {len(test_keys)}")

        # 检查重叠
        errors = []

        train_val_overlap = train_keys & val_keys
        if len(train_val_overlap) == 0:
            print("\n[OK] Train/Val: No overlap")
        else:
            print(f"\n[FAIL] Train/Val overlap: {len(train_val_overlap)}")
            for key in list(train_val_overlap)[:5]:
                print(f"  - {key}")
            errors.append(f"Train/Val overlap: {len(train_val_overlap)}")

        train_test_overlap = train_keys & test_keys
        if len(train_test_overlap) == 0:
            print("[OK] Train/Test: No overlap")
        else:
            print(f"[FAIL] Train/Test overlap: {len(train_test_overlap)}")
            for key in list(train_test_overlap)[:5]:
                print(f"  - {key}")
            errors.append(f"Train/Test overlap: {len(train_test_overlap)}")

        val_test_overlap = val_keys & test_keys
        if len(val_test_overlap) == 0:
            print("[OK] Val/Test: No overlap")
        else:
            print(f"[FAIL] Val/Test overlap: {len(val_test_overlap)}")
            for key in list(val_test_overlap)[:5]:
                print(f"  - {key}")
            errors.append(f"Val/Test overlap: {len(val_test_overlap)}")

        # 检查内部重复
        print("\n内部重复检查:")
        if len(train_list) == len(train_keys):
            print("[OK] Train: No internal duplicates")
        else:
            print(f"[FAIL] Train internal duplicates: {len(train_list) - len(train_keys)}")
            errors.append(f"Train internal duplicates: {len(train_list) - len(train_keys)}")

        if len(val_list) == len(val_keys):
            print("[OK] Val: No internal duplicates")
        else:
            print(f"[FAIL] Val internal duplicates: {len(val_list) - len(val_keys)}")
            errors.append(f"Val internal duplicates: {len(val_list) - len(val_keys)}")

        if len(test_list) == len(test_keys):
            print("[OK] Test: No internal duplicates")
        else:
            print(f"[FAIL] Test internal duplicates: {len(test_list) - len(test_keys)}")
            errors.append(f"Test internal duplicates: {len(test_list) - len(test_keys)}")

        # Subset 分布
        print_separator("Subset 分布")

        def count_subsets(item_list):
            counts = {}
            for item in item_list:
                if isinstance(item, dict):
                    subset = item['subset']
                else:
                    # 旧格式：从 ID 解析
                    subset = item.split('_')[0] if '_' in item else 'unknown'
                counts[subset] = counts.get(subset, 0) + 1
            return counts

        for split_name, item_list in [('Train', train_list), ('Val', val_list), ('Test', test_list)]:
            counts = count_subsets(item_list)
            print(f"\n{split_name}:")
            for subset in ['RSHaze_G', 'RSHaze_L', 'RSHaze_S']:
                print(f"  {subset}: {counts.get(subset, 0)}")

        # 汇总
        if len(errors) == 0:
            print("\n[OK] Split JSON 验证通过！")
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


def print_json_schema(split_file: str = 'experiments/haze_density/rshazeplus_split.json'):
    """打印 JSON schema"""
    print_separator("JSON Schema")

    split_path = Path(split_file)
    if not split_path.exists():
        print(f"[FAIL] Split file not found: {split_file}")
        return

    try:
        with open(split_path, 'r', encoding='utf-8') as f:
            split_data = json.load(f)

        print("\n顶层 keys:")
        for key in split_data.keys():
            val = split_data[key]
            if isinstance(val, list):
                print(f"  {key}: list[{len(val)}]")
                if len(val) > 0:
                    print(f"    样本示例：{val[0]}")
            elif isinstance(val, dict):
                print(f"  {key}: dict")
                for subkey, subval in val.items():
                    print(f"    {subkey}: {subval}")
            else:
                print(f"  {key}: {val}")

    except Exception as e:
        print(f"[FAIL] {e}")


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("RSHaze+ Split 验证 (修复版)")
    print("=" * 60)

    # 打印 schema
    print_json_schema()

    # 验证
    success = verify_split_json()

    print_separator("验证完成")

    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
