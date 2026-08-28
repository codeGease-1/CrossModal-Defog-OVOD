#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
完整雾密度网络测试脚本

【在 Colab 执行】

测试内容:
1. Shape Test - 输入输出形状
2. Range Test - 输出范围 [0, 1]
3. Finite Test - 无 NaN/Inf
4. Forward Test - 前向传播
5. Backward Test - 反向传播
6. GPU Test - GPU 执行
7. Performance Test - 参数量、显存、时间

使用方法:
    !python scripts/test_model.py
"""

import sys
from pathlib import Path
import time

# 设置路径
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import torch
from src.models.haze_density import (
    HazeDensityNet,
    generate_s_final,
)
from src.models.haze_density.haze_density_net import get_model_summary


def print_separator(title: str):
    """打印分隔线"""
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def count_parameters(module: torch.nn.Module) -> int:
    """计算参数量"""
    return sum(p.numel() for p in module.parameters())


def test_shape():
    """1. Shape Test"""
    print_separator("1. Shape Test")

    model = HazeDensityNet(base_channels=32)

    # 测试不同输入尺寸
    test_cases = [
        (1, 3, 256, 256),
        (2, 3, 128, 128),
        (4, 3, 512, 512),
    ]

    all_passed = True

    for B, C, H, W in test_cases:
        x = torch.rand(B, C, H, W)
        out = model(x)

        expected_shape = (B, 1, H, W)
        passed = out.shape == torch.Size(expected_shape)
        status = "[OK]" if passed else "[FAIL]"
        print(f"  {status} {tuple(x.shape)} -> {tuple(out.shape)} (expected {expected_shape})")

        if not passed:
            all_passed = False

    if all_passed:
        print("[OK] Shape test passed!")
    else:
        print("[FAIL] Shape test failed!")

    return all_passed


def test_range():
    """2. Range Test"""
    print_separator("2. Range Test")

    model = HazeDensityNet(base_channels=32)
    x = torch.rand(2, 3, 256, 256)
    out = model(x)

    # 检查输出范围 [0, 1]
    min_val = out.min().item()
    max_val = out.max().item()
    passed = min_val >= 0 and max_val <= 1

    status = "[OK]" if passed else "[FAIL]"
    print(f"  {status} Output range: [{min_val:.6f}, {max_val:.6f}]")

    if passed:
        print("[OK] Range test passed!")
    else:
        print("[FAIL] Range test failed!")

    return passed


def test_finite():
    """3. Finite Test"""
    print_separator("3. Finite Test")

    model = HazeDensityNet(base_channels=32)
    x = torch.rand(2, 3, 256, 256)
    out = model(x)

    passed = torch.isfinite(out).all()
    status = "[OK]" if passed else "[FAIL]"
    print(f"  {status} No NaN/Inf in output")

    if passed:
        print("[OK] Finite test passed!")
    else:
        print("[FAIL] Finite test failed!")

    return passed


def test_forward():
    """4. Forward Test"""
    print_separator("4. Forward Test")

    model = HazeDensityNet(base_channels=32)
    model.eval()

    x = torch.rand(2, 3, 256, 256)

    try:
        with torch.no_grad():
            out = model(x)

        passed = out.shape == (2, 1, 256, 256)
        status = "[OK]" if passed else "[FAIL]"
        print(f"  {status} Forward completed: {tuple(x.shape)} -> {tuple(out.shape)}")

        if passed:
            print("[OK] Forward test passed!")
        else:
            print("[FAIL] Forward test failed!")

        return passed
    except Exception as e:
        print(f"  [FAIL] Forward error: {e}")
        return False


def test_backward():
    """5. Backward Test"""
    print_separator("5. Backward Test")

    model = HazeDensityNet(base_channels=32)
    model.train()

    x = torch.rand(2, 3, 256, 256, requires_grad=True)

    try:
        out = model(x)
        loss = out.mean()
        loss.backward()

        passed = x.grad is not None and torch.isfinite(x.grad).all()
        status = "[OK]" if passed else "[FAIL]"
        print(f"  {status} Backward completed, gradient computed")

        if passed:
            print("[OK] Backward test passed!")
        else:
            print("[FAIL] Backward test failed!")

        return passed
    except Exception as e:
        print(f"  [FAIL] Backward error: {e}")
        return False


def test_gpu():
    """6. GPU Test"""
    print_separator("6. GPU Test")

    if not torch.cuda.is_available():
        print("[SKIP] CUDA not available")
        return True

    device = torch.device("cuda")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")

    model = HazeDensityNet(base_channels=32).to(device)
    x = torch.rand(2, 3, 256, 256, device=device)

    try:
        out = model(x)
        passed = out.device == device
        status = "[OK]" if passed else "[FAIL]"
        print(f"  {status} Output on GPU: {out.device}")

        if passed:
            print("[OK] GPU test passed!")
        else:
            print("[FAIL] GPU test failed!")

        return passed
    except Exception as e:
        print(f"  [FAIL] GPU test error: {e}")
        return False


def test_performance():
    """7. Performance Test"""
    print_separator("7. Performance Test")

    if not torch.cuda.is_available():
        print("[SKIP] CUDA not available, running on CPU")
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # 创建模型
    model = HazeDensityNet(base_channels=32).to(device)
    model.eval()

    # 参数量
    params = count_parameters(model)
    print(f"\n  Parameters: {params:,}")

    # 各模块参数量
    stats = model.get_parameter_stats()
    print(f"\n  Parameter breakdown:")
    print(f"    Encoder:    {stats['encoder']:,} ({stats['encoder']/stats['total']*100:.1f}%)")
    print(f"    MultiScale: {stats['multiscale']:,} ({stats['multiscale']/stats['total']*100:.1f}%)")
    print(f"    Fusion:     {stats['fusion']:,} ({stats['fusion']/stats['total']*100:.1f}%)")
    print(f"    Decoder:    {stats['decoder']:,} ({stats['decoder']/stats['total']*100:.1f}%)")

    # 测试输入
    x = torch.rand(1, 3, 256, 256, device=device)

    # 显存占用（仅 GPU）
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # Forward 时间
    num_warmup = 3
    num_tests = 10

    # Warmup
    for _ in range(num_warmup):
        with torch.no_grad():
            _ = model(x)

    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.time()
    with torch.no_grad():
        for _ in range(num_tests):
            out = model(x)

    if device.type == "cuda":
        torch.cuda.synchronize()
    end = time.time()

    forward_time = (end - start) / num_tests * 1000  # ms
    print(f"\n  Forward time: {forward_time:.2f} ms (avg over {num_tests} runs)")

    # 显存占用（仅 GPU）
    if device.type == "cuda":
        allocated = torch.cuda.memory_allocated(0) / 1024**2
        peak = torch.cuda.max_memory_allocated(0) / 1024**2
        print(f"  GPU memory allocated: {allocated:.2f} MB")
        print(f"  GPU memory peak: {peak:.2f} MB")

    # 打印模型摘要
    print("\n" + get_model_summary(model))

    print("[OK] Performance test completed!")
    return True


def test_full_pipeline():
    """8. Full Pipeline Test"""
    print_separator("8. Full Pipeline Test (Model + Physical Prior)")

    if not torch.cuda.is_available():
        print("[SKIP] CUDA not available")
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # 创建模型
    model = HazeDensityNet(base_channels=32).to(device)
    model.train()

    # 测试输入
    B, H, W = 2, 256, 256
    image = torch.rand(B, 3, H, W, device=device, requires_grad=True)

    try:
        # 计算物理先验（监督信号）
        target = generate_s_final(image)  # [B, 1, H, W]

        # 模型预测
        pred = model(image)  # [B, 1, H, W]

        # 检查形状
        shape_ok = pred.shape == target.shape
        status = "[OK]" if shape_ok else "[FAIL]"
        print(f"  {status} Shape match: pred={tuple(pred.shape)}, target={tuple(target.shape)}")

        # 计算损失（MSE）
        criterion = torch.nn.MSELoss()
        loss = criterion(pred, target)

        # 反向传播
        loss.backward()

        grad_ok = image.grad is not None and torch.isfinite(image.grad).all()
        status = "[OK]" if grad_ok else "[FAIL]"
        print(f"  {status} Backward completed, loss={loss.item():.6f}")

        all_ok = shape_ok and grad_ok

        if all_ok:
            print("[OK] Full pipeline test passed!")
        else:
            print("[FAIL] Full pipeline test failed!")

        return all_ok
    except Exception as e:
        print(f"  [FAIL] Full pipeline error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("雾密度感知网络完整测试")
    print("=" * 60)

    # 检查设备
    if torch.cuda.is_available():
        print(f"\nGPU available: {torch.cuda.get_device_name(0)}")
        print(f"PyTorch version: {torch.__version__}")
    else:
        print("\nGPU available: No (running on CPU)")
        print(f"PyTorch version: {torch.__version__}")

    # 运行测试
    results = []

    results.append(("Shape", test_shape()))
    results.append(("Range", test_range()))
    results.append(("Finite", test_finite()))
    results.append(("Forward", test_forward()))
    results.append(("Backward", test_backward()))
    results.append(("GPU", test_gpu()))
    results.append(("Performance", test_performance()))
    results.append(("Full Pipeline", test_full_pipeline()))

    # 汇总
    print_separator("测试结果汇总")

    for name, passed in results:
        status = "[OK]" if passed else "[FAIL]"
        print(f"{status} {name}")

    all_passed = all(r[1] for r in results)

    if all_passed:
        print("\n[OK] 所有测试通过！")
        print("\n下一步:")
        print("  1. 进入下一阶段：训练框架实现")
        print("  2. 创建 Dataset / DataLoader")
        print("  3. 实现训练循环")
    else:
        print("\n[FAIL] 部分测试未通过，请修复后重试")

    print("=" * 60)

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
