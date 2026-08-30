"""
数据模块

提供雾密度网络训练所需的数据加载功能。

子模块:
    - datasets: 含雾图像数据集
    - transforms: 数据增强变换
"""

import torch
from torch.utils.data import DataLoader

from .datasets import HazeDensityDataset, RSHazePlusDataset
from .transforms import (
    HazeTrainTransform,
    HazeValTransform,
    create_train_transform,
    create_val_transform,
)


def build_rshazeplus_dataloader(
    root: str,
    split: str = 'train',
    subsets: tuple = None,
    image_size: int = 256,
    batch_size: int = 4,
    num_workers: int = 2,
    pin_memory: bool = True,
    return_clean: bool = False,
    val_ratio: float = 0.1,
    split_file: str = None,
    shuffle: bool = None,
):
    """
    构建 RSHaze+ DataLoader

    Args:
        root: 数据集根目录
        split: 数据划分 (train/val/test)
        subsets: 使用的子集 (None 则使用全部)
        image_size: 输出图像尺寸
        batch_size: 批次大小
        num_workers: DataLoader worker 数量
        pin_memory: 是否使用 pinned memory
        return_clean: 是否返回 clear image
        val_ratio: val 比例
        split_file: split 文件路径
        shuffle: 是否打乱 (None 则 train=True, 其他=False)

    Returns:
        DataLoader
    """
    if subsets is None:
        subsets = ('RSHaze_G', 'RSHaze_L', 'RSHaze_S')

    if shuffle is None:
        shuffle = (split == 'train')

    dataset = HazeDensityDataset(
        root=root,
        split=split,
        subsets=subsets,
        image_size=image_size,
        return_clean=return_clean,
        val_ratio=val_ratio,
        split_file=split_file,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=(split == 'train'),
    )

    return dataloader


__all__ = [
    # Datasets
    "HazeDensityDataset",
    "RSHazePlusDataset",
    # Transforms
    "HazeTrainTransform",
    "HazeValTransform",
    "create_train_transform",
    "create_val_transform",
    # DataLoader
    "build_rshazeplus_dataloader",
]
