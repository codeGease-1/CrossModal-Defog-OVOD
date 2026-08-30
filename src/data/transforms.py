# -*- coding: utf-8 -*-
"""
数据增强变换模块

为雾密度感知网络提供专门的数据增强变换。

重要原则:
1. 只使用几何增强 (crop/flip)，不改变物理雾特性
2. 遥感图像允许垂直翻转 (工程假设)
3. 不使用颜色抖动等改变雾物理特性的增强
4. Normalization 仅使用 ToTensor ([0,255] -> [0,1])，不使用 ImageNet mean/std

Transform 流程:
    读取 RGB [H, W, 3] [0, 255]
        ↓
    ToTensor [3, H, W] [0, 1]
        ↓
    几何增强 (RandomCrop, Flip)
        ↓
    输出 [3, 256, 256] [0, 1]
"""

import torch
from torchvision import transforms
from typing import Optional, Tuple


class HazeTrainTransform:
    """
    训练数据增强变换

    包含:
    - RandomCrop: 随机裁剪到指定尺寸
    - HorizontalFlip: 水平翻转 (p=0.5)
    - VerticalFlip: 垂直翻转 (p=0.5, 遥感图像特有)
    - ToTensor: 转换为 tensor 并归一化到 [0, 1]

    注意:
    - ToTensor 在最后执行，确保增强操作在 PIL Image 上进行
    - 不使用颜色相关的增强 (ColorJitter 等)
    """

    def __init__(
        self,
        image_size: int = 256,
        hflip_prob: float = 0.5,
        vflip_prob: float = 0.5,
    ):
        """
        Args:
            image_size: 输出图像尺寸 (正方形)
            hflip_prob: 水平翻转概率
            vflip_prob: 垂直翻转概率 (遥感图像特有)
        """
        self.image_size = image_size
        self.hflip_prob = hflip_prob
        self.vflip_prob = vflip_prob

        # 定义变换序列 (ToTensor 在最后)
        self.transform = transforms.Compose([
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(p=hflip_prob),
            transforms.RandomVerticalFlip(p=vflip_prob),
            transforms.ToTensor(),  # [H, W, 3] [0,255] -> [3, H, W] [0, 1]
        ])

    def __call__(self, img):
        """
        Args:
            img: PIL Image [H, W, 3] [0, 255]

        Returns:
            tensor: [3, image_size, image_size] [0, 1]
        """
        return self.transform(img)


class HazeValTransform:
    """
    验证/测试数据变换

    包含:
    - Resize: 调整到指定尺寸 (双线性插值)
    - CenterCrop: 中心裁剪 (如果图像大于目标尺寸)
    - ToTensor: 转换为 tensor 并归一化到 [0, 1]

    注意:
    - 不使用随机增强
    - 确定性变换，确保结果可复现
    """

    def __init__(
        self,
        image_size: int = 256,
    ):
        """
        Args:
            image_size: 输出图像尺寸 (正方形)
        """
        self.image_size = image_size

        # 定义变换序列
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
        ])

    def __call__(self, img):
        """
        Args:
            img: PIL Image [H, W, 3] [0, 255]

        Returns:
            tensor: [3, image_size, image_size] [0, 1]
        """
        return self.transform(img)


def create_train_transform(
    image_size: int = 256,
    hflip_prob: float = 0.5,
    vflip_prob: float = 0.5,
) -> HazeTrainTransform:
    """
    创建训练变换

    Args:
        image_size: 输出图像尺寸
        hflip_prob: 水平翻转概率
        vflip_prob: 垂直翻转概率

    Returns:
        HazeTrainTransform 实例
    """
    return HazeTrainTransform(
        image_size=image_size,
        hflip_prob=hflip_prob,
        vflip_prob=vflip_prob,
    )


def create_val_transform(
    image_size: int = 256,
) -> HazeValTransform:
    """
    创建验证/测试变换

    Args:
        image_size: 输出图像尺寸

    Returns:
        HazeValTransform 实例
    """
    return HazeValTransform(image_size=image_size)
