#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据集验证脚本

【在 Colab 执行】

功能:
1. 统计文件数量
2. 检查图片格式
3. 检查 RGB 通道
4. 检查图像尺寸分布
5. 检测损坏文件
6. 显示随机样本
7. 检查 hazy/clear 配对关系
8. 输出数据集统计信息

使用方法:
    !python scripts/inspect_dataset.py --dataset RSHazePlus --data_dir /content/datasets/RSHazePlus
    !python scripts/inspect_dataset.py --dataset RRSHID --data_dir /content/datasets/RRSHID
    !python scripts/inspect_dataset.py --dataset RSHaze --data_dir /content/datasets/RSHaze
"""

import sys
import argparse
import os
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple, Optional

import torch
import torchvision
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


# 支持的图像格式
SUPPORTED_FORMATS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}


def find_image_files(directory: str, recursive: bool = True) -> List[Path]:
    """查找目录中的所有图像文件"""
    directory = Path(directory)
    if not directory.exists():
        print(f"  [WARN] Directory not found: {directory}")
        return []

    if recursive:
        files = [f for f in directory.rglob('*') if f.suffix.lower() in SUPPORTED_FORMATS]
    else:
        files = [f for f in directory.glob('*') if f.suffix.lower() in SUPPORTED_FORMATS]

    return files


def get_image_info(image_path: Path) -> Optional[Dict]:
    """获取单张图像的详细信息"""
    try:
        with Image.open(image_path) as img:
            return {
                'path': image_path,
                'width': img.width,
                'height': img.height,
                'mode': img.mode,
                'format': img.format,
                'valid': True,
            }
    except Exception as e:
        return {
            'path': image_path,
            'width': 0,
            'height': 0,
            'mode': 'ERROR',
            'format': 'ERROR',
            'valid': False,
            'error': str(e),
        }


def check_image_pairing(hazy_files: List[Path], clear_files: List[Path],
                        pairing_rule: str = 'same_name') -> Dict:
    """
    检查 hazy/clear 图像配对关系

    pairing_rule:
        - 'same_name': 文件名完全相同 (除了目录)
        - 'hazy_suffix': hazy 文件有 _haze 后缀
        - 'clear_suffix': clear 文件有 _clean 后缀
    """
    hazy_names = {f.name for f in hazy_files}
    clear_names = {f.name for f in clear_files}

    paired = hazy_names & clear_names
    hazy_only = hazy_names - clear_names
    clear_only = clear_names - hazy_names

    return {
        'total_hazy': len(hazy_files),
        'total_clear': len(clear_files),
        'paired_count': len(paired),
        'hazy_only_count': len(hazy_only),
        'clear_only_count': len(clear_only),
        'paired_names': list(paired)[:10],  # 前 10 个配对文件名
        'hazy_only_names': list(hazy_only)[:10],
        'clear_only_names': list(clear_only)[:10],
    }


def analyze_dataset(data_dir: str, dataset_name: str = 'Unknown') -> Dict:
    """
    分析数据集

    返回:
        dict: 包含统计信息的字典
    """
    data_dir = Path(data_dir)

    print(f"\n{'='*60}")
    print(f"数据集分析：{dataset_name}")
    print(f"目录：{data_dir}")
    print(f"{'='*60}\n")

    # 检查目录是否存在
    if not data_dir.exists():
        print(f"[ERROR] 目录不存在：{data_dir}")
        return {'error': f'Directory not found: {data_dir}'}

    # 1. 目录结构分析
    print("1. 目录结构分析")
    print("-" * 40)
    subdirs = [d.name for d in data_dir.iterdir() if d.is_dir()]
    print(f"  子目录：{subdirs}")

    # 2. 查找图像文件
    print("\n2. 图像文件统计")
    print("-" * 40)

    all_files = find_image_files(str(data_dir), recursive=True)
    print(f"  总图像文件数：{len(all_files)}")

    # 按目录分类
    dir_counter = Counter()
    for f in all_files:
        parent_name = f.parent.name
        dir_counter[parent_name] += 1

    print(f"  各目录文件数:")
    for dir_name, count in dir_counter.most_common():
        print(f"    {dir_name}: {count}")

    # 3. 图像格式分析
    print("\n3. 图像格式分析")
    print("-" * 40)

    format_counter = Counter()
    mode_counter = Counter()
    resolution_counter = Counter()
    valid_count = 0
    invalid_files = []

    for f in all_files[:100]:  # 抽样 100 张
        info = get_image_info(f)
        format_counter[info['format']] += 1
        mode_counter[info['mode']] += 1
        resolution_counter[(info['width'], info['height'])] += 1

        if info['valid']:
            valid_count += 1
        else:
            invalid_files.append(info)

    print(f"  格式分布:")
    for fmt, count in format_counter.most_common():
        print(f"    {fmt}: {count}")

    print(f"  颜色模式分布:")
    for mode, count in mode_counter.most_common():
        print(f"    {mode}: {count}")

    print(f"  分辨率分布 (前 10):")
    for (w, h), count in resolution_counter.most_common(10):
        print(f"    {w}x{h}: {count}")

    print(f"  抽样验证：{valid_count}/100 有效")

    if invalid_files:
        print(f"  [WARN] 发现损坏文件:")
        for info in invalid_files[:5]:
            print(f"    {info['path']}: {info.get('error', 'Unknown error')}")

    # 4. RGB 检查
    print("\n4. RGB 通道检查")
    print("-" * 40)

    rgb_count = sum(count for mode, count in mode_counter.items() if mode in ['RGB', 'RGBA'])
    grayscale_count = sum(count for mode, count in mode_counter.items() if mode in ['L', '1'])
    other_count = len(mode_counter) - len([m for m in mode_counter if m in ['RGB', 'RGBA', 'L', '1']])

    print(f"  RGB/RGBA: {rgb_count}")
    print(f"  Grayscale: {grayscale_count}")
    print(f"  Other: {other_count}")

    if rgb_count > 0:
        print(f"  [OK] 数据集包含 RGB 图像")
    else:
        print(f"  [WARN] 数据集可能不包含 RGB 图像")

    # 5. 检查 hazy/clear 配对
    print("\n5. Hazy/Clear 配对检查")
    print("-" * 40)

    # 尝试识别 hazy 和 clear 目录
    hazy_dirs = [d for d in subdirs if 'haze' in d.lower() or 'hazy' in d.lower()]
    clear_dirs = [d for d in subdirs if 'clear' in d.lower() or 'clean' in d.lower()]

    print(f"  可能的 Hazy 目录：{hazy_dirs}")
    print(f"  可能的 Clear 目录：{clear_dirs}")

    if hazy_dirs and clear_dirs:
        hazy_dir = data_dir / hazy_dirs[0]
        clear_dir = data_dir / clear_dirs[0]

        hazy_files = find_image_files(str(hazy_dir))
        clear_files = find_image_files(str(clear_dir))

        pairing_info = check_image_pairing(hazy_files, clear_files)

        print(f"  Hazy 文件数：{pairing_info['total_hazy']}")
        print(f"  Clear 文件数：{pairing_info['total_clear']}")
        print(f"  配对数量：{pairing_info['paired_count']}")
        print(f"  仅 Hazy: {pairing_info['hazy_only_count']}")
        print(f"  仅 Clear: {pairing_info['clear_only_count']}")

        if pairing_info['paired_count'] > 0:
            print(f"  [OK] 发现配对关系")
            print(f"  示例配对：{pairing_info['paired_names'][:3]}")
        else:
            print(f"  [WARN] 未发现明显配对关系")
    else:
        print(f"  [INFO] 无法自动识别 hazy/clear 目录结构")

    # 6. 检查官方 train/test split
    print("\n6. 官方 Train/Test Split 检查")
    print("-" * 40)

    split_dirs = [d for d in subdirs if d.lower() in ['train', 'val', 'test', 'validation']]
    print(f"  发现的 Split 目录：{split_dirs}")

    if split_dirs:
        for split_dir in split_dirs:
            split_path = data_dir / split_dir
            files = find_image_files(str(split_path))
            print(f"    {split_dir}: {len(files)} 文件")
        print(f"  [OK] 发现官方 split 划分")
    else:
        print(f"  [INFO] 未发现官方 split 划分，需要手动划分")

    # 7. 重复图像检查 (基于文件名)
    print("\n7. 重复图像检查")
    print("-" * 40)

    name_counter = Counter(f.name for f in all_files)
    duplicates = [name for name, count in name_counter.items() if count > 1]

    if duplicates:
        print(f"  [WARN] 发现重复文件名：{len(duplicates)} 个")
        for name in duplicates[:5]:
            count = name_counter[name]
            print(f"    {name}: {count} 次")
    else:
        print(f"  [OK] 未发现重复文件名")

    # 8. Patch Crop 适用性分析
    print("\n8. Patch Crop 适用性分析")
    print("-" * 40)

    resolutions = [(w, h) for (w, h), _ in resolution_counter.items()]
    if resolutions:
        min_res = min(resolutions, key=lambda x: x[0] * x[1])
        max_res = max(resolutions, key=lambda x: x[0] * x[1])

        print(f"  最小分辨率：{min_res[0]}x{min_res[1]}")
        print(f"  最大分辨率：{max_res[0]}x{max_res[1]}")

        if min_res[0] >= 256 and min_res[1] >= 256:
            print(f"  [OK] 适合 256x256 patch crop")
        if min_res[0] >= 512 and min_res[1] >= 512:
            print(f"  [OK] 适合 512x512 patch crop")
        else:
            print(f"  [WARN] 部分图像可能不适合 patch crop")
    else:
        print(f"  [INFO] 无法分析分辨率")

    # 9. 显示随机样本
    print("\n9. 随机样本预览")
    print("-" * 40)

    valid_files = [f for f in all_files[:200] if get_image_info(f)['valid']]

    if valid_files:
        # 显示 4 张随机样本
        sample_files = valid_files[:4]

        fig, axes = plt.subplots(1, len(sample_files), figsize=(20, 5))
        if len(sample_files) == 1:
            axes = [axes]

        for i, f in enumerate(sample_files):
            try:
                img = Image.open(f).convert('RGB')
                axes[i].imshow(img)
                axes[i].set_title(f"{f.name}\n{img.width}x{img.height}")
                axes[i].axis('off')
            except Exception as e:
                print(f"  [WARN] 无法显示 {f}: {e}")

        plt.tight_layout()
        save_path = data_dir.parent / f"{dataset_name}_preview.png"
        plt.savefig(str(save_path), dpi=100, bbox_inches='tight')
        print(f"  [OK] 预览图已保存：{save_path}")
        plt.close()
    else:
        print(f"  [INFO] 无可用图像显示")

    # 10. 总结
    print(f"\n{'='*60}")
    print("数据集分析总结")
    print(f"{'='*60}")

    summary = {
        'dataset_name': dataset_name,
        'data_dir': str(data_dir),
        'total_files': len(all_files),
        'subdirs': subdirs,
        'formats': dict(format_counter),
        'modes': dict(mode_counter),
        'resolutions': dict(resolution_counter.most_common(10)),
        'has_rgb': rgb_count > 0,
        'has_pairing': len(hazy_dirs) > 0 and len(clear_dirs) > 0,
        'has_official_split': len(split_dirs) > 0,
        'has_duplicates': len(duplicates) > 0,
        'valid_for_patch_crop': len(resolutions) > 0 and min(resolutions, key=lambda x: x[0] * x[1])[0] >= 256,
    }

    print(f"  总文件数：{summary['total_files']}")
    print(f"  包含 RGB: {'是' if summary['has_rgb'] else '否'}")
    print(f"  包含配对：{'是' if summary['has_pairing'] else '否'}")
    print(f"  官方 Split: {'是' if summary['has_official_split'] else '否'}")
    print(f"  重复文件：{'是' if summary['has_duplicates'] else '否'}")
    print(f"  适合 Patch Crop: {'是' if summary['valid_for_patch_crop'] else '否'}")

    print(f"\n{'='*60}\n")

    return summary


def main():
    parser = argparse.ArgumentParser(description='数据集验证脚本')
    parser.add_argument('--dataset', type=str, default='Unknown',
                        help='数据集名称 (RSHazePlus, RRSHID, RSHaze)')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='数据集目录路径')
    parser.add_argument('--output', type=str, default=None,
                        help='输出统计信息 JSON 文件路径')

    args = parser.parse_args()

    # 分析数据集
    summary = analyze_dataset(args.data_dir, args.dataset)

    # 保存统计信息
    if args.output:
        import json
        with open(args.output, 'w', encoding='utf-8') as f:
            # 转换 Path 对象为字符串
            summary_serializable = {
                k: str(v) if isinstance(v, Path) else v
                for k, v in summary.items()
            }
            json.dump(summary_serializable, f, indent=2, ensure_ascii=False)
        print(f"[OK] 统计信息已保存：{args.output}")

    return summary


if __name__ == '__main__':
    main()
