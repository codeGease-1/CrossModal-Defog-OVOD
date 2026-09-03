#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RSHaze+ 数据集结构分析脚本

【在 Colab 或本地执行】

功能:
1. 完整目录树统计
2. 各子目录文件数量
3. 图像格式/分辨率/颜色模式分析
4. RGB vs NIR 区分
5. clean/hazy 配对验证
6. 文件名规律分析

使用方法:
    python scripts/analyze_rshazeplus_structure.py --root datasets/RSHaze+
"""

import argparse
from pathlib import Path
from collections import defaultdict
from PIL import Image


def print_separator(title: str):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def analyze_directory_structure(root: Path):
    """分析完整目录结构"""
    print_separator(f"目录结构分析：{root}")

    structure = defaultdict(lambda: defaultdict(int))

    for path in root.rglob("*.png"):
        # 获取相对路径
        rel_path = path.relative_to(root)
        parts = rel_path.parts

        # 统计各级目录
        if len(parts) >= 2:
            level1 = parts[0]  # RSHaze_G/L/S/SOTS
            level2 = parts[1]  # train/test
            level3 = parts[2]  # cleanpng/synhazypng/etc

            structure[level1][level2] += 1
            structure[f"{level1}/{level2}/{level3}"]['files'] += 1

    # 输出
    for level1 in sorted(structure.keys()):
        if '/' not in level1:
            print(f"\n{level1}/")
            for level2 in sorted(structure[level1].keys()):
                if '/' not in level2:
                    count = structure[level1][level2]
                    print(f"  {level2}/: {count} PNG files")


def analyze_subdirectory_details(root: Path):
    """详细分析每个子目录"""
    print_separator("子目录详细分析")

    top_dirs = ['RSHaze_G', 'RSHaze_L', 'RSHaze_S', 'SOTS']

    for top_dir in top_dirs:
        dir_path = root / top_dir
        if not dir_path.exists():
            print(f"\n[{top_dir}] 目录不存在")
            continue

        print(f"\n{'='*40}")
        print(f"{top_dir}")
        print(f"{'='*40}")

        # 统计 train/test
        for split in ['train', 'test']:
            split_path = dir_path / split
            if not split_path.exists():
                continue

            print(f"\n  {split}/")
            for subdir in split_path.iterdir():
                if not subdir.is_dir():
                    continue

                png_files = list(subdir.glob("*.png"))
                print(f"    {subdir.name}/: {len(png_files)} PNG files")

                # 抽样检查图像属性
                if png_files:
                    sample = png_files[0]
                    try:
                        with Image.open(sample) as img:
                            print(f"      Sample: {sample.name}")
                            print(f"      Mode: {img.mode}, Size: {img.size}")
                    except Exception as e:
                        print(f"      Error: {e}")


def analyze_image_properties(root: Path):
    """分析图像属性分布"""
    print_separator("图像属性分析")

    stats = defaultdict(lambda: {'count': 0, 'modes': defaultdict(int), 'sizes': defaultdict(int)})

    for png_file in root.rglob("*.png"):
        try:
            with Image.open(png_file) as img:
                rel_path = png_file.relative_to(root)
                parts = rel_path.parts

                if len(parts) >= 2:
                    level1 = parts[0]
                    stats[level1]['count'] += 1
                    stats[level1]['modes'][img.mode] += 1
                    stats[level1]['sizes'][img.size] += 1
        except Exception:
            pass

    # 输出
    for level1 in sorted(stats.keys()):
        s = stats[level1]
        print(f"\n{level1}:")
        print(f"  Total PNG: {s['count']}")
        print(f"  Modes: {dict(s['modes'])}")
        print(f"  Sizes (top 5): {dict(list(s['sizes'].items())[:5])}")


def identify_rgb_vs_nir(root: Path):
    """区分 RGB 和 NIR 数据"""
    print_separator("RGB vs NIR 区分")

    nir_keywords = ['nir', 'NIR']

    for top_dir in ['RSHaze_G', 'RSHaze_L', 'RSHaze_S', 'SOTS']:
        dir_path = root / top_dir
        if not dir_path.exists():
            continue

        print(f"\n{top_dir}/")

        rgb_dirs = []
        nir_dirs = []

        for subdir in dir_path.rglob("*"):
            if not subdir.is_dir():
                continue

            subdir_name = subdir.name.lower()

            if any(kw in subdir_name for kw in nir_keywords):
                nir_dirs.append(str(subdir.relative_to(dir_path)))
            elif any(x in subdir_name for x in ['cleanpng', 'synhazypng', 'airpng', 'transpng']):
                rgb_dirs.append(str(subdir.relative_to(dir_path)))

        print(f"  RGB directories: {rgb_dirs}")
        print(f"  NIR directories: {nir_dirs}")


def verify_pairing(root: Path):
    """验证 clean/hazy 配对"""
    print_separator("Clean/Hazy 配对验证")

    for top_dir in ['RSHaze_G', 'RSHaze_L', 'RSHaze_S']:
        dir_path = root / top_dir
        if not dir_path.exists():
            continue

        print(f"\n{top_dir}/")

        for split in ['train', 'test']:
            split_path = dir_path / split
            if not split_path.exists():
                continue

            clean_dir = split_path / 'cleanpng'
            hazy_dir = split_path / 'synhazypng'

            if not clean_dir.exists() or not hazy_dir.exists():
                print(f"  {split}/: cleanpng or synhazypng missing")
                continue

            clean_names = {f.name for f in clean_dir.glob("*.png")}
            hazy_names = {f.name for f in hazy_dir.glob("*.png")}

            paired = clean_names & hazy_names
            clean_only = clean_names - hazy_names
            hazy_only = hazy_names - clean_names

            print(f"  {split}/:")
            print(f"    Clean only: {len(clean_only)}")
            print(f"    Hazy only: {len(hazy_only)}")
            print(f"    Paired: {len(paired)}")

            if paired:
                print(f"    Sample paired files: {list(paired)[:5]}")


def analyze_filename_pattern(root: Path):
    """分析文件名规律"""
    print_separator("文件名规律分析")

    for top_dir in ['RSHaze_G', 'RSHaze_L', 'RSHaze_S']:
        dir_path = root / top_dir
        if not dir_path.exists():
            continue

        print(f"\n{top_dir}/train/cleanpng:")

        clean_dir = dir_path / 'train' / 'cleanpng'
        if not clean_dir.exists():
            continue

        files = sorted(list(clean_dir.glob("*.png")))[:10]
        for f in files:
            print(f"  {f.name}")


def main():
    parser = argparse.ArgumentParser(description='RSHaze+ 数据集结构分析')
    parser.add_argument('--root', type=str, required=True, help='RSHaze+ 根目录')
    args = parser.parse_args()

    root = Path(args.root)

    if not root.exists():
        print(f"Error: Directory not found: {root}")
        return

    print(f"\nRSHaze+ 数据集结构分析")
    print(f"Root: {root}")

    # 执行分析
    analyze_directory_structure(root)
    analyze_subdirectory_details(root)
    analyze_image_properties(root)
    identify_rgb_vs_nir(root)
    verify_pairing(root)
    analyze_filename_pattern(root)

    print_separator("分析完成")


if __name__ == "__main__":
    main()
