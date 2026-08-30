#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RSHaze+ Dataset 可视化脚本 (Stage 5B-1)

随机选择 G/L/S 各若干张，生成预览图像。

输出:
    experiments/haze_density/results/dataset_preview/
    - hazy_samples.png
    - hazy_clean_pairs.png (如果 return_clean=True)
"""

import sys
from pathlib import Path

# 设置路径
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import random
import torch
from PIL import Image
import torchvision.utils as vutils

from src.data import HazeDensityDataset


def print_separator(title: str):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def create_grid_image(tensor_list: list, nrow: int = 4, padding: int = 2) -> Image:
    """
    创建网格图像

    Args:
        tensor_list: [3, H, W] tensors
        nrow: 每行数量
        padding: 间距

    Returns:
        PIL Image
    """
    # Make contiguous
    tensor_list = [t.contiguous() for t in tensor_list]

    # Create grid
    grid = vutils.make_grid(tensor_list, nrow=nrow, padding=padding, normalize=False)

    # Convert to PIL Image
    # grid: [3, H, W] [0, 1] -> [H, W, 3] [0, 255]
    numpy_image = grid.permute(1, 2, 0).mul(255).clamp(0, 255).byte().numpy()
    pil_image = Image.fromarray(numpy_image.astype('uint8'), mode='RGB')

    return pil_image


def visualize_dataset(
    root: str = 'datasets/RSHaze+',
    samples_per_subset: int = 4,
    image_size: int = 256,
    output_dir: str = 'experiments/haze_density/results/dataset_preview',
):
    """
    可视化数据集

    Args:
        root: 数据集根目录
        samples_per_subset: 每个 subset 采样的数量
        image_size: 图像尺寸
        output_dir: 输出目录
    """
    print_separator("RSHaze+ Dataset 可视化")

    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 加载数据集
    print("\n加载数据集...")

    train_ds = HazeDensityDataset(
        root=root,
        split='train',
        image_size=image_size,
        return_clean=True,
    )

    test_ds = HazeDensityDataset(
        root=root,
        split='test',
        image_size=image_size,
        return_clean=True,
    )

    print(f"Train: {len(train_ds)} samples")
    print(f"Test: {len(test_ds)} samples")

    # 按 subset 采样
    subsets = ['RSHaze_G', 'RSHaze_L', 'RSHaze_S']

    print(f"\n采样样本 (每个 subset {samples_per_subset} 个)...")

    hazy_samples = []
    clean_samples = []
    sample_info = []

    for subset in subsets:
        # 获取该 subset 的样本索引
        subset_indices = [
            i for i in range(len(train_ds))
            if train_ds[i]['subset'] == subset
        ]

        print(f"  {subset}: {len(subset_indices)} available")

        if len(subset_indices) == 0:
            print(f"    [WARN] No samples found in {subset}")
            continue

        # 随机采样
        selected_indices = random.sample(
            subset_indices,
            min(samples_per_subset, len(subset_indices))
        )

        for idx in selected_indices:
            sample = train_ds[idx]
            hazy_samples.append(sample['image'])
            if 'clean' in sample:
                clean_samples.append(sample['clean'])
            sample_info.append((subset, sample['id']))

    print(f"\n总采样数：{len(hazy_samples)}")

    # 保存 hazy 样本网格
    if len(hazy_samples) > 0:
        hazy_grid = create_grid_image(hazy_samples, nrow=samples_per_subset)
        hazy_path = output_path / 'hazy_samples.png'
        hazy_grid.save(hazy_path, quality=95)
        print(f"\n保存 hazy samples: {hazy_path}")

    # 保存 hazy-clean 配对网格
    if len(clean_samples) > 0:
        # 创建配对图像：[hazy1, clean1, hazy2, clean2, ...]
        paired_samples = []
        for i in range(len(hazy_samples)):
            paired_samples.append(hazy_samples[i])
            paired_samples.append(clean_samples[i])

        paired_grid = create_grid_image(paired_samples, nrow=samples_per_subset * 2)
        paired_path = output_path / 'hazy_clean_pairs.png'
        paired_grid.save(paired_path, quality=95)
        print(f"保存 hazy-clean pairs: {paired_path}")

    # 保存样本信息
    info_path = output_path / 'sample_info.txt'
    with open(info_path, 'w', encoding='utf-8') as f:
        f.write("Sample Information\n")
        f.write("=" * 40 + "\n\n")
        for i, (subset, sample_id) in enumerate(sample_info):
            f.write(f"{i+1}. {subset}: {sample_id}\n")
    print(f"保存样本信息：{info_path}")

    # 打印样本信息
    print("\n样本列表:")
    for i, (subset, sample_id) in enumerate(sample_info):
        print(f"  {i+1}. {subset}: {sample_id}")

    # 保存测试集样本
    print("\n" + "-" * 40)
    print("采样测试集样本...")

    test_hazy_samples = []
    test_clean_samples = []
    test_sample_info = []

    for subset in subsets:
        subset_indices = [
            i for i in range(len(test_ds))
            if test_ds[i]['subset'] == subset
        ]

        print(f"  {subset}: {len(subset_indices)} available")

        if len(subset_indices) == 0:
            continue

        selected_indices = random.sample(
            subset_indices,
            min(samples_per_subset, len(subset_indices))
        )

        for idx in selected_indices:
            sample = test_ds[idx]
            test_hazy_samples.append(sample['image'])
            if 'clean' in sample:
                test_clean_samples.append(sample['clean'])
            test_sample_info.append((subset, sample['id']))

    if len(test_hazy_samples) > 0:
        test_hazy_grid = create_grid_image(test_hazy_samples, nrow=samples_per_subset)
        test_hazy_path = output_path / 'test_hazy_samples.png'
        test_hazy_grid.save(test_hazy_path, quality=95)
        print(f"\n保存 test hazy samples: {test_hazy_path}")

    if len(test_clean_samples) > 0:
        test_paired_samples = []
        for i in range(len(test_hazy_samples)):
            test_paired_samples.append(test_hazy_samples[i])
            test_paired_samples.append(test_clean_samples[i])

        test_paired_grid = create_grid_image(test_paired_samples, nrow=samples_per_subset * 2)
        test_paired_path = output_path / 'test_hazy_clean_pairs.png'
        test_paired_grid.save(test_paired_path, quality=95)
        print(f"保存 test hazy-clean pairs: {test_paired_path}")

    print_separator("可视化完成")
    print(f"\n输出目录：{output_path}")
    print("\n生成的文件:")
    for f in sorted(output_path.glob("*")):
        print(f"  - {f.name}")


def main():
    """主函数"""
    random.seed(42)

    visualize_dataset(
        root='datasets/RSHaze+',
        samples_per_subset=4,
        image_size=256,
        output_dir='experiments/haze_density/results/dataset_preview',
    )

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
