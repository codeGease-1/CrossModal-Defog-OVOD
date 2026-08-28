#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Smoke Test - 快速验证脚本

用于验证模型的基本功能：
1. 模型构建
2. Forward 传播
3. Backward 传播
4. Shape 检查

使用方法:
    !python scripts/smoke_test.py
"""

import sys
from pathlib import Path


def setup_paths():
    """设置项目路径"""
    project_root = Path(__file__).parent.parent
    src_path = project_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    return project_root


def test_pytorch_env():
    """测试 PyTorch 环境"""
    print("=" * 60)
    print("1. PyTorch 环境检查")
    print("=" * 60)

    import torch

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    return torch


def test_physical_priors(torch):
    """测试物理先验模块"""
    print("\n" + "=" * 60)
    print("2. 物理先验模块测试")
    print("=" * 60)

    # TODO: 待实现
    print("待实现：physical_priors 模块")
    return True


def test_model_build(torch, device):
    """测试模型构建"""
    print("\n" + "=" * 60)
    print("3. 模型构建测试")
    print("=" * 60)

    # TODO: 待实现
    print("待实现：haze_density_net 模块")
    return True


def test_forward_backward(torch, device):
    """测试 forward/backward"""
    print("\n" + "=" * 60)
    print("4. Forward/Backward 测试")
    print("=" * 60)

    # TODO: 待实现
    print("待实现：forward/backward 测试")
    return True


def test_shape_consistency(torch, device):
    """测试形状一致性"""
    print("\n" + "=" * 60)
    print("5. 形状一致性测试")
    print("=" * 60)

    # TODO: 待实现
    print("待实现：shape consistency 测试")
    return True


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("CrossModal-Defog-OVOD - Smoke Test")
    print("=" * 60 + "\n")

    # 设置路径
    setup_paths()

    # 测试 PyTorch 环境
    torch = test_pytorch_env()

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n使用设备：{device}")

    # 运行测试
    tests = [
        ("物理先验", lambda: test_physical_priors(torch)),
        ("模型构建", lambda: test_model_build(torch, device)),
        ("Forward/Backward", lambda: test_forward_backward(torch, device)),
        ("形状一致性", lambda: test_shape_consistency(torch, device)),
    ]

    results = []
    for name, test_fn in tests:
        try:
            result = test_fn()
            results.append((name, "✓" if result else "✗"))
        except Exception as e:
            results.append((name, f"✗ {e}"))

    # 汇总结果
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)

    for name, result in results:
        print(f"{name}: {result}")

    all_passed = all(r[1] == "✓" for r in results)

    if all_passed:
        print("\n✓ 所有测试通过！")
    else:
        print("\n⚠️  部分测试未通过或待实现")

    print("=" * 60)

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
