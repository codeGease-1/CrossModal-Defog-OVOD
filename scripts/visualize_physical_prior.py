#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
物理先验可视化脚本

【在 Colab 执行】

对于一张输入图像，生成并保存:
- Input (原始图像)
- Dark Channel
- Local Contrast
- Color Shift
- Weighted Fusion (S_hat)
- S_final (Guided Filter 后)

使用方法:
    # 使用示例图像
    !python scripts/visualize_physical_prior.py --image path/to/image.jpg

    # 如果没有图像，会生成一个测试图像
    !python scripts/visualize_physical_prior.py --generate-test
"""

import sys
import argparse
from pathlib import Path
from typing import Optional

# 设置路径：将项目根目录添加到 sys.path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import torch
import torch.nn.functional as F
from src.models.haze_density import (
    generate_s_final,
    dark_channel,
    local_contrast,
    color_shift,
)


def load_image(image_path: str, target_size: int = 256) -> torch.Tensor:
    """
    加载图像并预处理

    Args:
        image_path: 图像路径
        target_size: 目标尺寸（短边）

    Returns:
        tensor: [1, 3, H, W], 范围 [0, 1]
    """
    try:
        from PIL import Image
    except ImportError:
        print("Error: Pillow not installed. Run: pip install Pillow")
        sys.exit(1)

    image = Image.open(image_path).convert("RGB")

    # 调整大小（保持长宽比，短边为 target_size）
    w, h = image.size
    scale = target_size / min(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    image = image.resize((new_w, new_h), Image.LANCZOS)

    # 转换为 tensor
    tensor = torch.from_numpy(np.array(image).astype(np.float32) / 255.0)
    tensor = tensor.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]

    return tensor


def generate_test_image(
    size: int = 256,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    生成测试图像（模拟含雾场景）

    Args:
        size: 图像尺寸
        device: 设备

    Returns:
        tensor: [1, 3, H, W]
    """
    if device is None:
        device = torch.device("cpu")

    # 创建一个渐变背景
    y, x = torch.meshgrid(
        torch.linspace(0, 1, size, device=device),
        torch.linspace(0, 1, size, device=device),
        indexing="ij",
    )

    # RGB 渐变
    r = x
    g = y
    b = 1 - (x + y) / 2
    b = torch.clamp(b, 0, 1)

    image = torch.stack([r, g, b], dim=0).unsqueeze(0)  # [1, 3, H, W]

    # 添加一些"雾"效果（降低对比度）
    image = 0.5 * image + 0.3

    return image


def save_image(tensor: torch.Tensor, save_path: str):
    """
    保存 tensor 为图像

    Args:
        tensor: [1, 1, H, W] 或 [1, 3, H, W]
        save_path: 保存路径
    """
    import numpy as np

    # 移除 batch 维度
    if tensor.dim() == 4:
        tensor = tensor[0]

    # 如果是单通道，复制为三通道
    if tensor.dim() == 2 or tensor.shape[0] == 1:
        tensor = tensor.repeat(3, 1, 1)

    # 转置为 HWC
    tensor = tensor.permute(1, 2, 0)

    # 转换为 numpy
    np_img = tensor.cpu().clamp(0, 1).numpy()

    # 保存
    from PIL import Image

    Image.fromarray(np_img).save(save_path)
    print(f"  Saved: {save_path}")


def visualize(image_path: Optional[str] = None, generate_test: bool = False):
    """
    主可视化函数

    Args:
        image_path: 输入图像路径
        generate_test: 是否生成测试图像
    """
    import numpy as np

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加载或生成图像
    if generate_test:
        print("Generating test image...")
        image = generate_test_image(256, device)
    elif image_path:
        print(f"Loading image: {image_path}")
        image = load_image(image_path).to(device)
    else:
        print("Error: Please provide --image or --generate-test")
        sys.exit(1)

    print(f"Input shape: {tuple(image.shape)}")

    # 创建输出目录
    output_dir = project_root / "experiments" / "haze_density" / "results" / "physical_prior"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存输入图像
    save_image(image, output_dir / "input.jpg")

    # 计算物理先验
    print("Computing physical priors...")
    result = generate_s_final(image, return_intermediate=True)

    D_hat = result["D_hat"]
    C_hat = result["C_hat"]
    K_hat = result["K_hat"]
    S_hat = result["S_hat"]
    S_final = result["S_final"]

    # 保存各阶段结果
    save_image(D_hat, output_dir / "dark_channel.jpg")
    save_image(C_hat, output_dir / "local_contrast.jpg")
    save_image(K_hat, output_dir / "color_shift.jpg")
    save_image(S_hat, output_dir / "weighted_fusion_S_hat.jpg")
    save_image(S_final, output_dir / "S_final.jpg")

    # 打印统计信息
    print("\nStatistics:")
    print(f"  D_hat:  min={D_hat.min():.4f}, max={D_hat.max():.4f}, mean={D_hat.mean():.4f}")
    print(f"  C_hat:  min={C_hat.min():.4f}, max={C_hat.max():.4f}, mean={C_hat.mean():.4f}")
    print(f"  K_hat:  min={K_hat.min():.4f}, max={K_hat.max():.4f}, mean={K_hat.mean():.4f}")
    print(f"  S_hat:  min={S_hat.min():.4f}, max={S_hat.max():.4f}, mean={S_hat.mean():.4f}")
    print(f"  S_final: min={S_final.min():.4f}, max={S_final.max():.4f}, mean={S_final.mean():.4f}")

    print(f"\nOutput directory: {output_dir}")
    print("Visualization complete!")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Physical Prior Visualization")

    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Input image path",
    )

    parser.add_argument(
        "--generate-test",
        action="store_true",
        help="Generate a test image instead of loading from file",
    )

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("Physical Prior Visualization")
    print("=" * 60 + "\n")

    visualize(args.image, args.generate_test)


if __name__ == "__main__":
    main()
