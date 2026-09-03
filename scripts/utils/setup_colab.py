#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Colab 环境初始化脚本

在 Google Colab 中运行此脚本，自动完成环境配置和检查。

使用方法:
    !python scripts/setup_colab.py
"""

import os
import sys
from pathlib import Path


def setup_paths():
    """设置项目路径：将项目根目录添加到 sys.path"""
    # 检测是否在 Colab 环境中
    in_colab = "google.colab" in sys.modules

    if in_colab:
        # Colab 环境：假设项目在 /content/CrossModal-Defog-OVOD
        project_root = Path("/content/CrossModal-Defog-OVOD")
    else:
        # 本地环境：使用脚本所在目录的父目录
        project_root = Path(__file__).parent.parent

    # 将项目根目录添加到 sys.path，以便使用 from src.xxx import 导入
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    return project_root


def check_pytorch():
    """检查 PyTorch 环境"""
    try:
        import torch
        print("=" * 60)
        print("PyTorch 环境检查")
        print("=" * 60)
        print(f"✓ PyTorch version: {torch.__version__}")

        if torch.cuda.is_available():
            print(f"✓ CUDA available: True")
            print(f"✓ CUDA version: {torch.version.cuda}")
            print(f"✓ GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"  GPU {i}: {props.name}")
                print(f"    Memory: {props.total_memory / 1024**3:.2f} GB")
                print(f"    Compute Capability: {props.major}.{props.minor}")
        else:
            print("✗ CUDA available: False")
            print("  警告：未检测到 GPU，将在 CPU 上运行（速度较慢）")

        return True
    except ImportError:
        print("✗ PyTorch 未安装")
        print("  请运行：!pip install torch torchvision")
        return False


def check_dependencies():
    """检查其他依赖"""
    deps = {
        "torchvision": "torchvision",
        "opencv": "cv2",
        "numpy": "numpy",
        "yaml": "yaml",
        "tqdm": "tqdm",
    }

    print("\n" + "=" * 60)
    print("依赖检查")
    print("=" * 60)

    all_ok = True
    for name, module in deps.items():
        try:
            __import__(module)
            print(f"✓ {name}")
        except ImportError:
            print(f"✗ {name} - 未安装")
            all_ok = False

    return all_ok


def create_directories(project_root):
    """创建必要的目录"""
    dirs = [
        "experiments/haze_density/checkpoints",
        "experiments/haze_density/logs",
        "experiments/haze_density/output",
    ]

    print("\n" + "=" * 60)
    print("目录检查")
    print("=" * 60)

    for d in dirs:
        path = project_root / d
        path.mkdir(parents=True, exist_ok=True)
        print(f"✓ {d}")


def print_config_summary():
    """打印配置摘要"""
    try:
        import yaml

        config_path = Path(__file__).parent.parent / "configs" / "haze_density.yaml"
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)

            print("\n" + "=" * 60)
            print("配置摘要")
            print("=" * 60)
            print(f"项目：{config['project']['name']}")
            print(f"任务：{config['project']['task']}")
            print(f"设备：{config['runtime']['device']}")
            print(f"AMP: {config['runtime']['amp']}")
            print(f"图像尺寸：{config['data']['image_size']}")
            print(f"Batch Size: {config['data']['batch_size']}")
            print(f"基础通道数：{config['model']['base_channels']}")
            print(f"学习率：{config['train']['lr']}")
            print(f"训练轮数：{config['train']['epochs']}")
    except Exception as e:
        print(f"\n配置读取失败：{e}")


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("CrossModal-Defog-OVOD - Colab 环境初始化")
    print("=" * 60 + "\n")

    # 设置路径
    project_root = setup_paths()
    print(f"项目根目录：{project_root}\n")

    # 检查 PyTorch
    if not check_pytorch():
        print("\n⚠️  PyTorch 检查失败，请先安装 PyTorch")
        return False

    # 检查依赖
    if not check_dependencies():
        print("\n⚠️  部分依赖未安装，请运行：!pip install -r requirements.txt")
        return False

    # 创建目录
    create_directories(project_root)

    # 打印配置摘要
    print_config_summary()

    print("\n" + "=" * 60)
    print("✓ 环境初始化完成！")
    print("=" * 60)
    print("\n下一步:")
    print("  1. 运行 smoke test: !python scripts/smoke_test.py")
    print("  2. 开始训练：!python scripts/train_haze_density.py")
    print()

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
