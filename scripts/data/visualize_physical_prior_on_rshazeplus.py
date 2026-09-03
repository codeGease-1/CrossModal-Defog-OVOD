#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Physical Prior Visualization on RSHaze+ (Stage 5B-2)

生成 Hazy RGB + S_final 对比图。

输出格式:
    ┌─────────────┬─────────────┐
    │    Hazy     │   S_final   │
    ├─────────────┼─────────────┤
    │    Hazy     │   S_final   │
    └─────────────┴─────────────┘

保存:
    experiments/haze_density/results/physical_prior/
    - g_hazy_prior.png
    - l_hazy_prior.png
    - s_hazy_prior.png
    - sample_info.txt
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

from src.data import build_rshazeplus_dataloader
from src.models.haze_density.physical_prior import PhysicalPriorModule


def print_separator(title: str):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def create_hazy_prior_pair_grid(
    hazy_list: list,
    prior_list: list,
    nrow: int = 4,
    padding: int = 2,
) -> Image:
    """
    创建 Hazy + Prior 配对网格

    布局：[hazy1, prior1, hazy2, prior2, ...]

    Args:
        hazy_list: [3, H, W] tensors [0, 1]
        prior_list: [1, H, W] tensors [0, 1]
        nrow: 每行配对数量
        padding: 间距

    Returns:
        PIL Image
    """
    # 交替排列
    paired = []
    for i in range(len(hazy_list)):
        paired.append(hazy_list[i])  # [3, H, W]
        # Prior 转为 3 通道 (灰度)
        prior_3ch = torch.cat([prior_list[i]] * 3, dim=0)  # [3, H, W]
        paired.append(prior_3ch)

    # Make contiguous
    paired = [t.contiguous() for t in paired]

    # Create grid
    grid = vutils.make_grid(paired, nrow=nrow * 2, padding=padding, normalize=False)

    # Convert to PIL Image
    numpy_image = grid.permute(1, 2, 0).mul(255).clamp(0, 255).byte().numpy()
    pil_image = Image.fromarray(numpy_image.astype('uint8'), mode='RGB')

    return pil_image


def visualize_physical_prior(
    root: str = 'datasets/RSHaze+',
    samples_per_subset: int = 4,
    image_size: int = 256,
    output_dir: str = 'experiments/haze_density/results/physical_prior',
):
    """
    可视化 Physical Prior

    Args:
        root: 数据集根目录
        samples_per_subset: 每个 subset 采样数量
        image_size: 图像尺寸
        output_dir: 输出目录
    """
    print_separator("Physical Prior Visualization")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Physical Prior 模块
    physical_prior = PhysicalPriorModule(
        window_size=15,
        guided_radius=15,
        guided_eps=0.01,
    ).to(device)
    physical_prior.eval()

    # 按 subset 分别可视化
    subsets = ['RSHaze_G', 'RSHaze_L', 'RSHaze_S']
    subset_names = {'RSHaze_G': 'g', 'RSHaze_L': 'l', 'RSHaze_S': 's'}

    all_sample_info = []

    for subset in subsets:
        print(f"\nProcessing {subset}...")

        # 创建只包含该 subset 的 DataLoader
        train_loader = build_rshazeplus_dataloader(
            root=root,
            split='train',
            subsets=(subset,),
            image_size=image_size,
            batch_size=4,
            num_workers=0,
            split_file='experiments/haze_density/rshazeplus_split.json',
        )

        print(f"  {subset} loader: {len(train_loader)} batches")

        # 采样
        hazy_samples = []
        prior_samples = []
        sample_info = []

        count = 0
        for batch in train_loader:
            if count >= samples_per_subset:
                break

            image = batch['image'].to(device, non_blocking=True)
            filenames = batch['filename']

            with torch.no_grad():
                s_final = physical_prior(image)

            # 保存单个样本
            for i in range(image.shape[0]):
                if count >= samples_per_subset:
                    break

                hazy_samples.append(image[i].cpu())  # [3, H, W]
                prior_samples.append(s_final[i].cpu())  # [1, H, W]

                # 统计信息
                s = s_final[i].squeeze()
                info = {
                    'subset': subset,
                    'filename': filenames[i],
                    'shape': list(s_final[i].shape),
                    'min': s.min().item(),
                    'max': s.max().item(),
                    'mean': s.mean().item(),
                }
                sample_info.append(info)
                all_sample_info.append(info)

                print(f"  Sample {len(sample_info)}: {filenames[i]}, "
                      f"S_final range=[{s.min():.4f}, {s.max():.4f}], mean={s.mean():.4f}")

            count += 1

        if len(hazy_samples) == 0:
            print(f"  [WARN] No samples for {subset}")
            continue

        # 生成配对网格图
        grid_image = create_hazy_prior_pair_grid(
            hazy_samples,
            prior_samples,
            nrow=samples_per_subset,
        )

        # 保存
        output_file = output_path / f"{subset_names[subset]}_hazy_prior.png"
        grid_image.save(output_file, quality=95)
        print(f"  Saved: {output_file}")

    # 保存样本信息
    info_file = output_path / 'sample_info.txt'
    with open(info_file, 'w', encoding='utf-8') as f:
        f.write("Physical Prior Sample Information\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Image size: {image_size}x{image_size}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Physical Prior params: window_size=15, guided_radius=15, guided_eps=0.01\n")
        f.write("\n" + "-" * 60 + "\n\n")

        for i, info in enumerate(all_sample_info, 1):
            f.write(f"{i}. {info['subset']}: {info['filename']}\n")
            f.write(f"   Shape: {info['shape']}\n")
            f.write(f"   S_final min: {info['min']:.6f}\n")
            f.write(f"   S_final max: {info['max']:.6f}\n")
            f.write(f"   S_final mean: {info['mean']:.6f}\n\n")

    print(f"\nSaved sample info: {info_file}")

    # 打印汇总统计
    print_separator("Summary Statistics")

    for subset in subsets:
        subset_info = [s for s in all_sample_info if s['subset'] == subset]
        if len(subset_info) > 0:
            means = [s['mean'] for s in subset_info]
            mins = [s['min'] for s in subset_info]
            maxs = [s['max'] for s in subset_info]

            print(f"\n{subset}:")
            print(f"  Samples: {len(subset_info)}")
            print(f"  Mean range: [{min(means):.4f}, {max(means):.4f}]")
            print(f"  Min range: [{min(mins):.4f}, {max(mins):.4f}]")
            print(f"  Max range: [{min(maxs):.4f}, {max(maxs):.4f}]")

    print_separator("Visualization Complete")
    print(f"\nOutput directory: {output_path}")
    print("\nGenerated files:")
    for f in sorted(output_path.glob("*")):
        print(f"  - {f.name}")

    return True


def main():
    """主函数"""
    random.seed(42)

    visualize_physical_prior(
        root='datasets/RSHaze+',
        samples_per_subset=4,
        image_size=256,
        output_dir='experiments/haze_density/results/physical_prior',
    )

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
