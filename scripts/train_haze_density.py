#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
雾密度感知网络训练脚本

使用方法:
    # 基本训练
    python scripts/train_haze_density.py --config configs/haze_density.yaml

    # 自定义参数（使用下划线替代点号）
    python scripts/train_haze_density.py \
        --config configs/haze_density.yaml \
        --data_batch_size 8 \
        --train_lr 5e-4

    # 断点续训
    python scripts/train_haze_density.py \
        --config configs/haze_density.yaml \
        --resume experiments/haze_density/checkpoints/latest.pt
"""

import argparse
import sys
from pathlib import Path


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="Haze Density Network Training",
        allow_abbrev=False,  # 禁用参数缩写，避免解析冲突
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/haze_density.yaml",
        help="配置文件路径",
    )

    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="从 checkpoint 恢复训练",
    )

    # 允许通过命令行覆盖配置（使用下划线替代点号）
    parser.add_argument("--data_batch_size", type=int, default=None,
                        help="覆盖配置文件中的 data.batch_size")
    parser.add_argument("--data_image_size", type=int, default=None,
                        help="覆盖配置文件中的 data.image_size")
    parser.add_argument("--train_lr", type=float, default=None,
                        help="覆盖配置文件中的 train.lr")
    parser.add_argument("--train_epochs", type=int, default=None,
                        help="覆盖配置文件中的 train.epochs")
    parser.add_argument("--model_base_channels", type=int, default=None,
                        help="覆盖配置文件中的 model.base_channels")

    return parser.parse_args()


def load_config(config_path, args):
    """加载配置文件并应用命令行覆盖"""
    import yaml

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 应用命令行覆盖
    if args.data_batch_size is not None:
        config["data"]["batch_size"] = args.data_batch_size
    if args.data_image_size is not None:
        config["data"]["image_size"] = args.data_image_size
    if args.train_lr is not None:
        config["train"]["lr"] = args.train_lr
    if args.train_epochs is not None:
        config["train"]["epochs"] = args.train_epochs
    if args.model_base_channels is not None:
        config["model"]["base_channels"] = args.model_base_channels

    return config


def main():
    """主训练函数"""
    args = parse_args()

    # 检查配置
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"错误：配置文件不存在：{config_path}")
        sys.exit(1)

    # 加载配置
    config = load_config(config_path, args)

    print("=" * 60)
    print("雾密度感知网络训练")
    print("=" * 60)
    print(f"配置：{args.config}")
    print(f"设备：{config['runtime']['device']}")
    print(f"Batch Size: {config['data']['batch_size']}")
    print(f"图像尺寸：{config['data']['image_size']}")
    print(f"学习率：{config['train']['lr']}")
    print(f"训练轮数：{config['train']['epochs']}")
    print("=" * 60)

    # TODO: 实现训练逻辑
    # 1. 初始化模型
    # 2. 加载数据
    # 3. 设置优化器
    # 4. 训练循环
    # 5. 保存 checkpoint

    print("\n训练逻辑待实现...")
    print("请等待后续阶段完成模型和数据模块的实现。")


if __name__ == "__main__":
    main()
