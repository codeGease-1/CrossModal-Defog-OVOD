# -*- coding: utf-8 -*-
"""
RSHaze+ 数据集模块 (Stage 5B-1)

基于 RSHaze+ 实际目录结构实现。

目录结构:
    datasets/RSHaze+/
    ├── RSHaze_G/
    │   ├── train/
    │   │   ├── cleanpng/      # 1000
    │   │   ├── synhazypng/    # 1000
    │   │   ├── airpng/        # 1000 (排除)
    │   │   └── transpng/      # 1000 (排除)
    │   └── test/
    │       ├── cleanpng/      # 330
    │       └── synhazypng/    # 330
    ├── RSHaze_L/
    │   ├── train/
    │   │   ├── cleanpng/      # ~2700
    │   │   └── synhazypng/    # ~2700
    │   └── test/
    │       ├── cleanpng/      # 270
    │       └── synhazypng/    # 270
    ├── RSHaze_S/
    │   ├── train/
    │   │   ├── cleanpng/      # 1000
    │   │   ├── synhazypng/    # 1000
    │   │   └── (nir* 排除)
    │   └── test/
    │       ├── cleanpng/      # 330
    │       └── synhazypng/    # 330
    └── SOTS/                  # 空目录 (排除)

配对规则:
    cleanpng/{id}.png ↔ synhazypng/{id}.png

重要原则:
1. 使用官方 train/test split
2. val 从 train 中按 90/10 划分 (seed=42)
3. 只使用 RGB (cleanpng/synhazypng)
4. 排除 NIR、airpng、transpng、SOTS
5. 按文件名精确配对
"""

from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple
from PIL import Image
import json
import hashlib
import random

from .transforms import create_train_transform, create_val_transform


# ============================================================================
# RSHaze+ Dataset
# ============================================================================

class RSHazePlusDataset:
    """
    RSHaze+ 数据集

    最终 Split (Stage 5B-1):
        | Subset   | Train | Val  | Test | Total |
        |----------|-------|------|------|-------|
        | RSHaze_G | 900   | 100  | 330  | 1330  |
        | RSHaze_L | 4374  | 486  | 270  | 5130  |
        | RSHaze_S | 900   | 100  | 330  | 1330  |
        | Total    | 6174  | 686  | 930  | 7790  |

    Args:
        root: 数据集根目录
        split: 数据划分 (train/val/test)
        subsets: 使用的子集 (默认全部)
        image_size: 输出图像尺寸
        transform: 数据变换
        return_clean: 是否返回 clear image
        val_ratio: val 从 train 划分的比例 (默认 0.1)
        split_file: split 文件路径 (可选，用于固定 split)
    """

    SUPPORTED_FORMATS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}
    ALL_SUBSETS = ['RSHaze_G', 'RSHaze_L', 'RSHaze_S']

    def __init__(
        self,
        root: str,
        split: str = 'train',
        subsets: Optional[Tuple[str, ...]] = None,
        image_size: int = 256,
        transform: Optional[Any] = None,
        return_clean: bool = False,
        val_ratio: float = 0.1,
        split_file: Optional[str] = None,
    ):
        """
        Args:
            root: 数据集根目录 (如 datasets/RSHaze+)
            split: 数据划分 (train/val/test)
            subsets: 使用的子集 (None 则使用全部)
            image_size: 输出图像尺寸
            transform: 数据变换
            return_clean: 是否返回 clear image
            val_ratio: val 比例 (仅当 split='val' 时)
            split_file: split 文件路径 (用于加载/保存 split)
        """
        self.root = Path(root)
        self.split = split
        self.subsets = subsets if subsets is not None else tuple(self.ALL_SUBSETS)
        self.image_size = image_size
        self.return_clean = return_clean
        self.val_ratio = val_ratio
        self.split_file = split_file

        # 加载图像列表
        self._load_image_list()

        # 处理 split
        self._apply_split()

        # 设置变换
        if transform is not None:
            self.transform = transform
        elif split == 'train':
            self.transform = create_train_transform(image_size)
        else:
            self.transform = create_val_transform(image_size)

        # 验证
        self._validate()

    def _load_image_list(self):
        """加载所有图像列表 (不区分 split)"""
        self._all_pairs = []

        for subset in self.subsets:
            subset_dir = self.root / subset

            # 检查 train 和 test
            for split_name in ['train', 'test']:
                split_dir = subset_dir / split_name
                if not split_dir.exists():
                    continue

                # hazy 和 clean 目录
                hazy_dir = split_dir / 'synhazypng'
                clean_dir = split_dir / 'cleanpng'

                if not hazy_dir.exists():
                    continue

                # 获取所有 hazy 文件
                hazy_files = sorted([
                    f for f in hazy_dir.iterdir()
                    if f.suffix.lower() in self.SUPPORTED_FORMATS and f.is_file()
                ])

                for hazy_path in hazy_files:
                    # 按文件名配对
                    clear_path = None
                    if clean_dir.exists():
                        candidate = clean_dir / hazy_path.name
                        if candidate.exists():
                            clear_path = candidate

                    # 生成唯一 ID
                    sample_id = self._generate_id(subset, split_name, hazy_path.name)

                    self._all_pairs.append({
                        'hazy_path': hazy_path,
                        'clear_path': clear_path,
                        'subset': subset,
                        'official_split': split_name,
                        'id': sample_id,
                        'filename': hazy_path.name,
                    })

        # 统计
        self._subset_counts = {}
        self._official_split_counts = {}
        for item in self._all_pairs:
            subset = item['subset']
            official_split = item['official_split']
            self._subset_counts[subset] = self._subset_counts.get(subset, 0) + 1
            self._official_split_counts[official_split] = self._official_split_counts.get(official_split, 0) + 1

    def _generate_id(self, subset: str, split: str, filename: str) -> str:
        """生成唯一 ID"""
        return f"{subset}_{split}_{Path(filename).stem}"

    def _apply_split(self):
        """应用 split"""
        if self.split == 'test':
            # 使用官方 test
            self.image_list = [
                item for item in self._all_pairs
                if item['official_split'] == 'test'
            ]
        elif self.split == 'train' or self.split == 'val':
            # 从官方 train 中划分 train/val
            train_items = [
                item for item in self._all_pairs
                if item['official_split'] == 'train'
            ]

            if self.split_file:
                # 尝试加载保存的 split
                split_file = Path(self.split_file)
                if split_file.exists():
                    with open(split_file, 'r', encoding='utf-8') as f:
                        split_data = json.load(f)

                    # 新 schema: [{"subset": "...", "filename": "..."}]
                    if isinstance(split_data.get('val'), list) and len(split_data['val']) > 0:
                        if isinstance(split_data['val'][0], dict):
                            # 新格式
                            val_keys = set(
                                (item['subset'], item['filename'])
                                for item in split_data['val']
                            )
                            train_keys = set(
                                (item['subset'], item['filename'])
                                for item in split_data['train']
                            )

                            if self.split == 'val':
                                self.image_list = [
                                    item for item in train_items
                                    if (item['subset'], item['filename']) in val_keys
                                ]
                                return
                            elif self.split == 'train':
                                self.image_list = [
                                    item for item in train_items
                                    if (item['subset'], item['filename']) in train_keys
                                ]
                                return
                    else:
                        # 旧格式或其他格式，打印警告
                        print(f"[WARN] Split file {split_file} has unexpected format, using default split")
                else:
                    print(f"[WARN] Split file {split_file} not found, using default split")

            # 按 subset 分别划分，保持分布
            random.seed(42)

            if self.split == 'val':
                val_keys = set()  # 使用 (subset, filename) 作为键
                for subset in self.subsets:
                    subset_items = [
                        item for item in train_items
                        if item['subset'] == subset
                    ]
                    random.shuffle(subset_items)
                    n_val = max(1, int(len(subset_items) * self.val_ratio))
                    for item in subset_items[:n_val]:
                        val_keys.add((item['subset'], item['filename']))

                self.image_list = [
                    item for item in train_items
                    if (item['subset'], item['filename']) in val_keys
                ]

                # 保存 split (如果是 val 且指定了 split_file)
                if self.split_file:
                    val_list = [
                        {'subset': item['subset'], 'filename': item['filename']}
                        for item in self.image_list
                    ]
                    train_list = [
                        {'subset': item['subset'], 'filename': item['filename']}
                        for item in train_items
                        if (item['subset'], item['filename']) not in val_keys
                    ]
                    test_list = [
                        {'subset': item['subset'], 'filename': item['filename']}
                        for item in self._all_pairs
                        if item['official_split'] == 'test'
                    ]
                    split_data = {
                        'train': train_list,
                        'val': val_list,
                        'test': test_list,
                        'metadata': {
                            'val_ratio': self.val_ratio,
                            'seed': 42,
                        }
                    }
                    split_file.parent.mkdir(parents=True, exist_ok=True)
                    with open(split_file, 'w', encoding='utf-8') as f:
                        json.dump(split_data, f, indent=2, ensure_ascii=False)
            else:
                # train
                self.image_list = train_items
        else:
            raise ValueError(f"Unknown split: {self.split}")

    def _validate(self):
        """验证数据集"""
        if len(self.image_list) == 0:
            raise ValueError(
                f"No images found for split='{self.split}' in subsets={self.subsets}. "
                f"Check if directory exists: {self.root}"
            )

        # 检查配对
        missing_clear = sum(1 for item in self.image_list if item['clear_path'] is None)
        if missing_clear > 0:
            print(f"[WARN] {missing_clear} samples missing clear image")

        # 检查文件存在性
        for item in self.image_list[:5]:
            if not item['hazy_path'].exists():
                raise ValueError(f"Hazy image not found: {item['hazy_path']}")

    def __len__(self) -> int:
        return len(self.image_list)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Args:
            idx: 图像索引

        Returns:
            dict 包含:
                - image: hazy image tensor [3, H, W] [0, 1]
                - subset: 子集名称 (RSHaze_G/L/S)
                - id: 样本 ID
                - path: 图像路径
                - clean: clear image tensor (如果 return_clean=True)
        """
        item = self.image_list[idx]

        # 加载 hazy 图像
        hazy_img = Image.open(item['hazy_path']).convert('RGB')
        hazy_tensor = self.transform(hazy_img)

        result = {
            'image': hazy_tensor,
            'subset': item['subset'],
            'filename': item['filename'],
            'id': item['id'],
            'path': str(item['hazy_path']),
        }

        if self.return_clean and item['clear_path'] is not None:
            clear_img = Image.open(item['clear_path']).convert('RGB')
            clear_tensor = self.transform(clear_img)
            result['clean'] = clear_tensor

        return result

    def get_stats(self) -> Dict[str, Any]:
        """获取数据集统计信息"""
        stats = {
            'dataset': 'RSHaze+',
            'split': self.split,
            'total_samples': len(self.image_list),
            'subsets': list(self._subset_counts.keys()),
            'subset_counts': dict(self._subset_counts),
            'image_size': self.image_size,
            'root': str(self.root),
        }

        # 当前 split 的 subset 分布
        current_subset_counts = {}
        for item in self.image_list:
            subset = item['subset']
            current_subset_counts[subset] = current_subset_counts.get(subset, 0) + 1
        stats['current_subset_counts'] = current_subset_counts

        return stats

    def get_all_ids(self) -> List[str]:
        """获取所有样本 ID"""
        return [item['id'] for item in self.image_list]

    def get_subset_samples(self, subset: str) -> List[Dict[str, Any]]:
        """获取指定子集的样本"""
        return [item for item in self.image_list if item['subset'] == subset]


# ============================================================================
# 统一接口
# ============================================================================

class HazeDensityDataset(RSHazePlusDataset):
    """
    雾密度数据集统一接口

    当前仅支持 RSHaze+。

    使用示例:
        # 训练集
        train_ds = HazeDensityDataset(
            root='datasets/RSHaze+',
            split='train',
            image_size=256,
        )

        # 验证集
        val_ds = HazeDensityDataset(
            root='datasets/RSHaze+',
            split='val',
            image_size=256,
        )

        # 测试集
        test_ds = HazeDensityDataset(
            root='datasets/RSHaze+',
            split='test',
            image_size=512,
        )

        # 仅使用特定子集
        train_ds = HazeDensityDataset(
            root='datasets/RSHaze+',
            split='train',
            subsets=('RSHaze_G', 'RSHaze_L'),
        )
    """

    def get_stats(self) -> Dict[str, Any]:
        stats = super().get_stats()
        stats['interface'] = 'HazeDensityDataset'
        return stats
