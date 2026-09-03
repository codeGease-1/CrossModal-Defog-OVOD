# -*- coding: utf-8 -*-
"""
数据集路径自动检测工具

支持自动检测以下环境：
1. Colab: /content/datasets/RSHazePlus/RSHaze+
2. Kaggle: /kaggle/input/datasets/iris114/rshaze/RSHaze+
3. 本地：datasets/RSHaze+

使用方法:
    from src.utils.path_utils import get_dataset_root

    # 自动检测
    dataset_root = get_dataset_root()

    # 或指定环境
    dataset_root = get_dataset_root(force_env='colab')
"""

import os
import sys
from pathlib import Path


# 各环境的数据集路径配置
DATASET_PATHS = {
    'colab': '/content/datasets/RSHazePlus/RSHaze+',
    'kaggle': '/kaggle/input/datasets/iris114/rshaze/RSHaze+',
    'local': 'datasets/RSHaze+',
}


def detect_environment() -> str:
    """
    自动检测当前运行环境

    检测优先级：
    1. Colab (通过多种检测方式)
    2. Kaggle (通过 kaggle 模块或环境变量)
    3. 本地

    Returns:
        环境标识：'colab' | 'kaggle' | 'local'
    """
    # ========== 检测 Colab ==========
    # 方式 1: 检查 google.colab 模块
    if 'google.colab' in sys.modules:
        print("[DEBUG] Detected Colab via google.colab module")
        return 'colab'

    # 方式 2: 尝试导入 google.colab.runtime
    try:
        import google.colab.runtime
        print("[DEBUG] Detected Colab via google.colab.runtime")
        return 'colab'
    except ImportError:
        pass

    # 方式 3: 检查 /content 目录（Colab 特有）
    if os.path.exists('/content') and not os.path.exists('/kaggle'):
        print("[DEBUG] Detected Colab via /content directory")
        return 'colab'

    # 方式 4: 检查 COLAB_RELEASE 环境变量
    if os.environ.get('COLAB_RELEASE'):
        print("[DEBUG] Detected Colab via COLAB_RELEASE env")
        return 'colab'

    # ========== 检测 Kaggle ==========
    # 方式 1: 检查 kaggle 模块
    if 'kaggle' in sys.modules:
        print("[DEBUG] Detected Kaggle via kaggle module")
        return 'kaggle'

    # 方式 2: 检查 Kaggle 特有的环境变量
    if os.environ.get('KAGGLE_KERNEL_RUN_TYPE'):
        print("[DEBUG] Detected Kaggle via KAGGLE_KERNEL_RUN_TYPE env")
        return 'kaggle'

    # 方式 3: 检查 /kaggle 目录（Kaggle 特有）
    if os.path.exists('/kaggle'):
        print("[DEBUG] Detected Kaggle via /kaggle directory")
        return 'kaggle'

    # ========== 默认本地 ==========
    print("[DEBUG] Detected local environment")
    return 'local'


def get_dataset_root(force_env: str = None) -> str:
    """
    获取数据集根路径

    Args:
        force_env: 强制指定环境 (None 则自动检测)

    Returns:
        数据集路径字符串

    Raises:
        RuntimeError: 如果指定的路径不存在
    """
    import sys

    # 确定环境
    if force_env:
        env = force_env.lower()
        if env not in DATASET_PATHS:
            raise ValueError(f"未知环境：{force_env}. 支持：{list(DATASET_PATHS.keys())}")
    else:
        env = detect_environment()

    # 获取路径
    dataset_root = DATASET_PATHS[env]

    # 验证路径存在
    if not os.path.exists(dataset_root):
        # 尝试给出更友好的错误信息
        available_paths = []
        for name, path in DATASET_PATHS.items():
            if os.path.exists(path):
                available_paths.append(f"  - {name}: {path}")

        if available_paths:
            print(f"[WARN] 指定路径不存在：{dataset_root}")
            print(f"[INFO] 可用的数据集路径:")
            for p in available_paths:
                print(p)
        else:
            print(f"[ERROR] 未找到任何数据集路径")
            print(f"[INFO] 检查以下路径:")
            for name, path in DATASET_PATHS.items():
                print(f"  - {name}: {path}")

        raise RuntimeError(f"数据集路径不存在：{dataset_root}")

    print(f"[INFO] 检测到环境：{env}")
    print(f"[INFO] 数据集路径：{dataset_root}")

    return dataset_root


def get_split_file_path() -> str:
    """
    获取 split 文件路径

    Returns:
        split 文件路径
    """
    return 'experiments/haze_density/rshazeplus_split.json'


def get_checkpoint_dir() -> str:
    """
    获取 checkpoint 目录

    Returns:
        checkpoint 目录路径
    """
    return 'experiments/haze_density/checkpoints/formal'


def get_result_dir() -> str:
    """
    获取结果目录

    Returns:
        结果目录路径
    """
    return 'experiments/haze_density/results'


# 便捷函数
def get_all_paths():
    """
    获取所有路径配置

    Returns:
        dict 包含所有路径
    """
    return {
        'dataset_root': get_dataset_root(),
        'split_file': get_split_file_path(),
        'checkpoint_dir': get_checkpoint_dir(),
        'result_dir': get_result_dir(),
    }


if __name__ == '__main__':
    # 测试
    print("路径检测测试")
    print("=" * 60)

    env = detect_environment()
    print(f"当前环境：{env}")

    paths = get_all_paths()
    for name, path in paths.items():
        exists = "✓" if os.path.exists(path) else "✗"
        print(f"{exists} {name}: {path}")
