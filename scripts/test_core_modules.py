#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
雾密度网络核心模块测试脚本

【在 Colab 执行】

测试内容:
1. test_encoder - Encoder 模块测试
2. test_rb - ResidualBlock 测试
3. test_sdrb - DilatedResidualBlock 测试
4. test_eca - ECA 模块测试
5. test_multiscale - 多尺度分支测试

每个测试检查:
- shape
- dtype
- device
- finite
- backward (梯度)

使用方法:
    !python scripts/test_core_modules.py
"""

import sys
from pathlib import Path

# 设置路径
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import torch
from src.models.haze_density import (
    Encoder,
    ResidualBlock,
    DilatedResidualBlock,
    ECA,
    ParallelMultiScaleFeatureExtractor,
)


def print_separator(title: str):
    """打印分隔线"""
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def count_parameters(module: torch.nn.Module) -> int:
    """计算参数量"""
    return sum(p.numel() for p in module.parameters())


def test_encoder():
    """1. Encoder 测试"""
    print_separator("1. Encoder Test")

    base_channels = 32
    encoder = Encoder(base_channels=base_channels)

    # 测试输入
    B, C, H, W = 2, 3, 256, 256
    x = torch.rand(B, C, H, W, requires_grad=True)

    # Forward
    out = encoder(x)

    # 检查形状
    expected_shape = (B, base_channels * 2, H // 2, W // 2)
    shape_ok = out.shape == torch.Size(expected_shape)
    status = "[OK]" if shape_ok else "[FAIL]"
    print(f"  {status} Shape: {tuple(x.shape)} -> {tuple(out.shape)} (expected {expected_shape})")

    # 检查 dtype
    dtype_ok = out.dtype == x.dtype
    status = "[OK]" if dtype_ok else "[FAIL]"
    print(f"  {status} dtype: {out.dtype}")

    # 检查 device
    device_ok = out.device == x.device
    status = "[OK]" if device_ok else "[FAIL]"
    print(f"  {status} device: {out.device}")

    # 检查 finite
    finite_ok = torch.isfinite(out).all()
    status = "[OK]" if finite_ok else "[FAIL]"
    print(f"  {status} finite: no NaN/Inf")

    # Backward 测试
    try:
        loss = out.mean()
        loss.backward()
        grad_ok = x.grad is not None and torch.isfinite(x.grad).all()
        status = "[OK]" if grad_ok else "[FAIL]"
        print(f"  {status} backward: gradient computed")
    except Exception as e:
        print(f"  [FAIL] backward error: {e}")
        grad_ok = False

    # 参数量
    params = count_parameters(encoder)
    print(f"  Parameters: {params:,}")

    all_ok = shape_ok and dtype_ok and device_ok and finite_ok and grad_ok
    if all_ok:
        print("[OK] Encoder test passed!")
    else:
        print("[FAIL] Encoder test failed!")

    return all_ok


def test_rb():
    """2. ResidualBlock 测试"""
    print_separator("2. ResidualBlock Test")

    channels = 64
    rb = ResidualBlock(channels=channels)

    # 测试输入
    B, C, H, W = 2, channels, 128, 128
    x = torch.rand(B, C, H, W, requires_grad=True)

    # Forward
    out = rb(x)

    # 检查形状（输入输出应一致）
    shape_ok = out.shape == x.shape
    status = "[OK]" if shape_ok else "[FAIL]"
    print(f"  {status} Shape: {tuple(x.shape)} -> {tuple(out.shape)}")

    # 检查 dtype
    dtype_ok = out.dtype == x.dtype
    status = "[OK]" if dtype_ok else "[FAIL]"
    print(f"  {status} dtype: {out.dtype}")

    # 检查 device
    device_ok = out.device == x.device
    status = "[OK]" if device_ok else "[FAIL]"
    print(f"  {status} device: {out.device}")

    # 检查 finite
    finite_ok = torch.isfinite(out).all()
    status = "[OK]" if finite_ok else "[FAIL]"
    print(f"  {status} finite: no NaN/Inf")

    # Backward 测试
    try:
        loss = out.mean()
        loss.backward()
        grad_ok = x.grad is not None and torch.isfinite(x.grad).all()
        status = "[OK]" if grad_ok else "[FAIL]"
        print(f"  {status} backward: gradient computed")
    except Exception as e:
        print(f"  [FAIL] backward error: {e}")
        grad_ok = False

    # 参数量
    params = count_parameters(rb)
    print(f"  Parameters: {params:,}")

    all_ok = shape_ok and dtype_ok and device_ok and finite_ok and grad_ok
    if all_ok:
        print("[OK] ResidualBlock test passed!")
    else:
        print("[FAIL] ResidualBlock test failed!")

    return all_ok


def test_sdrb():
    """3. DilatedResidualBlock 测试"""
    print_separator("3. DilatedResidualBlock Test")

    channels = 64
    all_ok = True

    # 测试不同 dilation rate（申报书规定：2, 3, 4）
    for dilation in [2, 3, 4]:
        print(f"\n  Testing dilation={dilation}")
        sdrb = DilatedResidualBlock(channels=channels, dilation=dilation)

        # 测试输入
        B, C, H, W = 2, channels, 128, 128
        x = torch.rand(B, C, H, W, requires_grad=True)

        # Forward
        out = sdrb(x)

        # 检查形状（输入输出应一致）
        shape_ok = out.shape == x.shape
        status = "[OK]" if shape_ok else "[FAIL]"
        print(f"    {status} Shape: {tuple(x.shape)} -> {tuple(out.shape)}")

        # 检查 finite
        finite_ok = torch.isfinite(out).all()
        status = "[OK]" if finite_ok else "[FAIL]"
        print(f"    {status} finite: no NaN/Inf")

        # Backward 测试
        try:
            loss = out.mean()
            loss.backward()
            grad_ok = x.grad is not None and torch.isfinite(x.grad).all()
            status = "[OK]" if grad_ok else "[FAIL]"
            print(f"    {status} backward: gradient computed")
        except Exception as e:
            print(f"    [FAIL] backward error: {e}")
            grad_ok = False

        if not (shape_ok and finite_ok and grad_ok):
            all_ok = False

    if all_ok:
        print("\n[OK] DilatedResidualBlock test passed!")
    else:
        print("\n[FAIL] DilatedResidualBlock test failed!")

    return all_ok


def test_eca():
    """4. ECA 测试"""
    print_separator("4. ECA Test")

    channels = 64
    eca = ECA(channels=channels)

    # 测试输入
    B, C, H, W = 2, channels, 128, 128
    x = torch.rand(B, C, H, W, requires_grad=True)

    # Forward
    out = eca(x)

    # 检查形状（输入输出应一致）
    shape_ok = out.shape == x.shape
    status = "[OK]" if shape_ok else "[FAIL]"
    print(f"  {status} Shape: {tuple(x.shape)} -> {tuple(out.shape)}")

    # 检查 dtype
    dtype_ok = out.dtype == x.dtype
    status = "[OK]" if dtype_ok else "[FAIL]"
    print(f"  {status} dtype: {out.dtype}")

    # 检查 device
    device_ok = out.device == x.device
    status = "[OK]" if device_ok else "[FAIL]"
    print(f"  {status} device: {out.device}")

    # 检查 finite
    finite_ok = torch.isfinite(out).all()
    status = "[OK]" if finite_ok else "[FAIL]"
    print(f"  {status} finite: no NaN/Inf")

    # 检查范围（sigmoid 输出应在 [0, 1]）
    range_ok = out.min() >= 0 and out.max() <= 1
    status = "[OK]" if range_ok else "[FAIL]"
    print(f"  {status} range: [{out.min():.4f}, {out.max():.4f}]")

    # Backward 测试
    try:
        loss = out.mean()
        loss.backward()
        grad_ok = x.grad is not None and torch.isfinite(x.grad).all()
        status = "[OK]" if grad_ok else "[FAIL]"
        print(f"  {status} backward: gradient computed")
    except Exception as e:
        print(f"  [FAIL] backward error: {e}")
        grad_ok = False

    # 参数量
    params = count_parameters(eca)
    print(f"  Parameters: {params:,}")

    all_ok = shape_ok and dtype_ok and device_ok and finite_ok and range_ok and grad_ok
    if all_ok:
        print("[OK] ECA test passed!")
    else:
        print("[FAIL] ECA test failed!")

    return all_ok


def test_multiscale():
    """5. MultiScale 测试"""
    print_separator("5. MultiScale Test")

    channels = 64
    multiscale = ParallelMultiScaleFeatureExtractor(channels=channels)

    # 测试输入
    B, C, H, W = 2, channels, 128, 128
    x = torch.rand(B, C, H, W, requires_grad=True)

    # Forward
    out = multiscale(x)

    # 检查形状（输出应为 3*C）
    expected_shape = (B, channels * 3, H, W)
    shape_ok = out.shape == torch.Size(expected_shape)
    status = "[OK]" if shape_ok else "[FAIL]"
    print(f"  {status} Shape: {tuple(x.shape)} -> {tuple(out.shape)} (expected {expected_shape})")

    # 检查 dtype
    dtype_ok = out.dtype == x.dtype
    status = "[OK]" if dtype_ok else "[FAIL]"
    print(f"  {status} dtype: {out.dtype}")

    # 检查 device
    device_ok = out.device == x.device
    status = "[OK]" if device_ok else "[FAIL]"
    print(f"  {status} device: {out.device}")

    # 检查 finite
    finite_ok = torch.isfinite(out).all()
    status = "[OK]" if finite_ok else "[FAIL]"
    print(f"  {status} finite: no NaN/Inf")

    # Backward 测试
    try:
        loss = out.mean()
        loss.backward()
        grad_ok = x.grad is not None and torch.isfinite(x.grad).all()
        status = "[OK]" if grad_ok else "[FAIL]"
        print(f"  {status} backward: gradient computed")
    except Exception as e:
        print(f"  [FAIL] backward error: {e}")
        grad_ok = False

    # 参数量
    params = count_parameters(multiscale)
    print(f"  Parameters: {params:,}")

    all_ok = shape_ok and dtype_ok and device_ok and finite_ok and grad_ok
    if all_ok:
        print("[OK] MultiScale test passed!")
    else:
        print("[FAIL] MultiScale test failed!")

    return all_ok


def test_gpu():
    """6. GPU 测试"""
    print_separator("6. GPU Test")

    if not torch.cuda.is_available():
        print("[SKIP] CUDA not available")
        return True

    device = torch.device("cuda")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")

    all_ok = True

    # 测试 Encoder
    print("  Testing Encoder on GPU")
    try:
        encoder = Encoder(base_channels=32).to(device)
        x = torch.rand(1, 3, 128, 128, device=device)
        out = encoder(x)
        ok = out.device == device
        status = "[OK]" if ok else "[FAIL]"
        print(f"    {status} Encoder on GPU")
        if not ok:
            all_ok = False
    except Exception as e:
        print(f"    [FAIL] Encoder on GPU: {e}")
        all_ok = False

    # 测试 MultiScale
    print("  Testing MultiScale on GPU")
    try:
        multiscale = ParallelMultiScaleFeatureExtractor(channels=64).to(device)
        x = torch.rand(1, 64, 64, 64, device=device)
        out = multiscale(x)
        ok = out.device == device
        status = "[OK]" if ok else "[FAIL]"
        print(f"    {status} MultiScale on GPU")
        if not ok:
            all_ok = False
    except Exception as e:
        print(f"    [FAIL] MultiScale on GPU: {e}")
        all_ok = False

    if all_ok:
        print("[OK] GPU test passed!")
    else:
        print("[FAIL] GPU test failed!")

    return all_ok


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("雾密度网络核心模块测试")
    print("=" * 60)

    # 检查设备
    if torch.cuda.is_available():
        print(f"\nGPU available: {torch.cuda.get_device_name(0)}")
    else:
        print("\nGPU available: No (running on CPU)")

    # 运行测试
    results = []

    results.append(("Encoder", test_encoder()))
    results.append(("ResidualBlock", test_rb()))
    results.append(("DilatedResidualBlock", test_sdrb()))
    results.append(("ECA", test_eca()))
    results.append(("MultiScale", test_multiscale()))
    results.append(("GPU", test_gpu()))

    # 汇总
    print_separator("测试结果汇总")

    for name, passed in results:
        status = "[OK]" if passed else "[FAIL]"
        print(f"{status} {name}")

    all_passed = all(r[1] for r in results)

    if all_passed:
        print("\n[OK] 所有测试通过！")
        print("\n下一步:")
        print("  1. 进入下一阶段：Decoder + 完整模型实现")
    else:
        print("\n[FAIL] 部分测试未通过，请修复后重试")

    print("=" * 60)

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
