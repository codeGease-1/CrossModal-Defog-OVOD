#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
通用 GPU 服务器 / 本地环境初始化脚本

从脚本位置定位项目，检查依赖与 CUDA，不安装或替换已有 PyTorch。

使用方法:
    python scripts/utils/setup_environment.py --require-cuda
"""

import argparse
import sys
from pathlib import Path


def setup_paths():
    """设置项目路径：将项目根目录添加到 sys.path"""
    project_root = Path(__file__).resolve().parents[2]

    # 将项目根目录添加到 sys.path，以便使用 from src.xxx import 导入
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    return project_root


def check_pytorch(require_cuda=False):
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
            if require_cuda:
                print("  GPU 检查失败：请确认平台已分配 GPU、驱动正常且安装的是 CUDA 版 PyTorch。")
                return False
            print("  CPU 模式：可用于小张量检查；云端训练请使用 --require-cuda。")

        if require_cuda:
            # 驱动可见还不足以证明 CUDA 运算可用。
            probe = torch.ones(1, device='cuda') * 2
            torch.cuda.synchronize()
            if probe.item() != 2:
                raise RuntimeError('CUDA tensor check failed')

        return True
    except (ImportError, OSError, RuntimeError) as error:
        print(f"✗ PyTorch/CUDA 检查失败：{error}")
        print("  请使用平台的 PyTorch GPU 镜像，或按 PyTorch 官方安装说明配置。")
        return False


def check_dependencies():
    """检查其他依赖"""
    deps = {
        "torchvision": "torchvision",
        "opencv": "cv2",
        "numpy": "numpy",
        "yaml": "yaml",
        "tqdm": "tqdm",
        "Pillow": "PIL",
        "matplotlib": "matplotlib",
        "scipy": "scipy",
    }

    print("\n" + "=" * 60)
    print("依赖检查")
    print("=" * 60)

    all_ok = True
    for name, module in deps.items():
        try:
            __import__(module)
            print(f"✓ {name}")
        except (ImportError, OSError, RuntimeError) as error:
            print(f"✗ {name} - 无法导入：{error}")
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

        config_path = Path(__file__).resolve().parents[2] / "configs" / "haze_density.yaml"
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)

            print("\n" + "=" * 60)
            print("配置摘要")
            print("=" * 60)
            for key in ('amp', 'image_size', 'batch_size', 'base_channels',
                        'lr', 'epochs', 'checkpoint_dir', 'result_dir'):
                print(f"{key}: {config[key]}")
    except Exception as e:
        print(f"\n配置读取失败：{e}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--require-cuda', action='store_true', help='没有可用 CUDA 时以失败退出')
    args = parser.parse_args()
    print("\n" + "=" * 60)
    print("CrossModal-Defog-OVOD - 通用环境初始化")
    print("=" * 60 + "\n")

    # 设置路径
    project_root = setup_paths()
    print(f"项目根目录：{project_root}\n")

    # 检查 PyTorch
    if not check_pytorch(require_cuda=args.require_cuda):
        print("\nPyTorch/CUDA 检查失败，请先修复上述环境问题")
        return False

    # 检查依赖
    if not check_dependencies():
        print("\n部分依赖无法导入，请检查环境并运行：python -m pip install -r requirements.txt")
        return False

    # 创建目录
    create_directories(project_root)

    # 打印配置摘要
    print_config_summary()

    print("\n" + "=" * 60)
    print("✓ 环境初始化完成！")
    print("=" * 60)
    print("\n下一步:")
    print("  1. 运行 smoke test: python scripts/utils/smoke_test.py")
    print("  2. 按 docs/雾密度网络_最简实验流程.md 准备数据、持久化实验配置并训练")
    print()

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
