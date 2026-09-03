#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
物理先验模块测试脚本

【在 Colab 执行】

测试内容:
1. Shape Test - 验证输出形状
2. Range Test - 验证输出范围 [0, 1]
3. Finite Test - 验证无 NaN/Inf
4. Batch Test - 验证不同 batch size
5. GPU Test - 验证 GPU 执行
6. Constructive Test - 构造性测试（常数图、渐变图、局部雾图）

使用方法:
    !python scripts/test_physical_prior.py
"""

import sys
from pathlib import Path

# 设置路径：将项目根目录添加到 sys.path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import torch
from src.models.haze_density import (
    dark_channel,
    local_contrast,
    color_shift,
    compute_physical_prior,
    generate_s_final,
    WEIGHT_DARK,
    WEIGHT_CONTRAST,
    WEIGHT_COLOR,
    EXPONENT_MU,
)


def print_separator(title: str):
    """打印分隔线"""
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def test_shapes():
    """1. Shape Test"""
    print_separator("1. Shape Test")

    # 测试不同输入尺寸
    test_cases = [
        (1, 3, 256, 256),
        (2, 3, 128, 128),
        (4, 3, 512, 512),
        (1, 3, 100, 200),  # 非正方形
    ]

    all_passed = True

    for B, C, H, W in test_cases:
        image = torch.rand(B, C, H, W)

        # 测试各模块输出形状
        d = dark_channel(image)
        c = local_contrast(image)
        k = color_shift(image)
        d_hat, c_hat, k_hat, s_hat = compute_physical_prior(image)
        s_final = generate_s_final(image)

        expected_shape = (B, 1, H, W)

        # 创建变量映射字典
        results = {
            "dark_channel": d,
            "local_contrast": c,
            "color_shift": k,
            "d_hat": d_hat,
            "c_hat": c_hat,
            "k_hat": k_hat,
            "s_hat": s_hat,
            "s_final": s_final,
        }

        checks = [
            ("dark_channel", d.shape == expected_shape),
            ("local_contrast", c.shape == expected_shape),
            ("color_shift", k.shape == expected_shape),
            ("d_hat", d_hat.shape == expected_shape),
            ("c_hat", c_hat.shape == expected_shape),
            ("k_hat", k_hat.shape == expected_shape),
            ("s_hat", s_hat.shape == expected_shape),
            ("s_final", s_final.shape == expected_shape),
        ]

        for name, passed in checks:
            status = "[OK]" if passed else "[FAIL]"
            print(f"  {status} {name}: {list(image.shape)} -> {list(results[name].shape)}")
            if not passed:
                all_passed = False

    if all_passed:
        print("[OK] All shape tests passed!")
    else:
        print("[FAIL] Some shape tests failed!")

    return all_passed


def test_range():
    """2. Range Test"""
    print_separator("2. Range Test")

    image = torch.rand(2, 3, 256, 256)

    d_hat, c_hat, k_hat, s_hat = compute_physical_prior(image)
    s_final = generate_s_final(image)

    all_passed = True

    # 创建变量映射字典
    results = {
        "d_hat": d_hat,
        "c_hat": c_hat,
        "k_hat": k_hat,
        "s_hat": s_hat,
        "s_final": s_final,
    }

    # 检查范围 [0, 1]
    checks = [
        ("d_hat", d_hat.min() >= 0 and d_hat.max() <= 1),
        ("c_hat", c_hat.min() >= 0 and c_hat.max() <= 1),
        ("k_hat", k_hat.min() >= 0 and k_hat.max() <= 1),
        ("s_hat", s_hat.min() >= 0 and s_hat.max() <= 1),
        ("s_final", s_final.min() >= 0 and s_final.max() <= 1),
    ]

    for name, passed in checks:
        tensor = results[name]
        status = "[OK]" if passed else "[FAIL]"
        print(f"  {status} {name}: range [{tensor.min():.4f}, {tensor.max():.4f}]")
        if not passed:
            all_passed = False

    if all_passed:
        print("[OK] All range tests passed!")
    else:
        print("[FAIL] Some range tests failed!")

    return all_passed


def test_finite():
    """3. Finite Test"""
    print_separator("3. Finite Test")

    # 测试正常图像
    image = torch.rand(2, 3, 256, 256)

    d_hat, c_hat, k_hat, s_hat = compute_physical_prior(image)
    s_final = generate_s_final(image)

    all_passed = True

    checks = [
        ("d_hat", torch.isfinite(d_hat).all()),
        ("c_hat", torch.isfinite(c_hat).all()),
        ("k_hat", torch.isfinite(k_hat).all()),
        ("s_hat", torch.isfinite(s_hat).all()),
        ("s_final", torch.isfinite(s_final).all()),
    ]

    for name, passed in checks:
        status = "[OK]" if passed else "[FAIL]"
        print(f"  {status} {name}: no NaN/Inf")
        if not passed:
            all_passed = False

    # 测试边界情况：全零图像
    print("  Testing edge case: all-zero image")
    image_zero = torch.zeros(1, 3, 256, 256)
    try:
        s_final_zero = generate_s_final(image_zero)
        passed = torch.isfinite(s_final_zero).all()
        status = "[OK]" if passed else "[FAIL]"
        print(f"  {status} all-zero image: no NaN/Inf")
        if not passed:
            all_passed = False
    except Exception as e:
        print(f"  [FAIL] all-zero image: {e}")
        all_passed = False

    # 测试边界情况：全一图像
    print("  Testing edge case: all-ones image")
    image_ones = torch.ones(1, 3, 256, 256)
    try:
        s_final_ones = generate_s_final(image_ones)
        passed = torch.isfinite(s_final_ones).all()
        status = "[OK]" if passed else "[FAIL]"
        print(f"  {status} all-ones image: no NaN/Inf")
        if not passed:
            all_passed = False
    except Exception as e:
        print(f"  [FAIL] all-ones image: {e}")
        all_passed = False

    if all_passed:
        print("[OK] All finite tests passed!")
    else:
        print("[FAIL] Some finite tests failed!")

    return all_passed


def test_batch():
    """4. Batch Test"""
    print_separator("4. Batch Test")

    batch_sizes = [1, 2, 4, 8]
    all_passed = True

    for B in batch_sizes:
        image = torch.rand(B, 3, 256, 256)
        s_final = generate_s_final(image)

        passed = s_final.shape == (B, 1, 256, 256)
        status = "[OK]" if passed else "[FAIL]"
        print(f"  {status} batch_size={B}: output shape {tuple(s_final.shape)}")
        if not passed:
            all_passed = False

    if all_passed:
        print("[OK] All batch tests passed!")
    else:
        print("[FAIL] Some batch tests failed!")

    return all_passed


def test_gpu():
    """5. GPU Test"""
    print_separator("5. GPU Test")

    if not torch.cuda.is_available():
        print("[SKIP] CUDA not available")
        return True

    device = torch.device("cuda")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")

    image = torch.rand(2, 3, 256, 256, device=device)

    try:
        s_final = generate_s_final(image)

        passed = s_final.device.type == "cuda"
        status = "[OK]" if passed else "[FAIL]"
        print(f"  {status} output on GPU: {s_final.device}")

        if passed:
            print("[OK] GPU test passed!")
        else:
            print("[FAIL] GPU test failed!")

        return passed

    except Exception as e:
        print(f"  [FAIL] GPU test error: {e}")
        return False


def test_constructive():
    """6. Constructive Test"""
    print_separator("6. Constructive Test")

    all_passed = True

    # 6.1 常数图像
    print("  6.1 Constant Image Test")
    image_const = torch.full((1, 3, 256, 256), 0.5)
    try:
        result = generate_s_final(image_const, return_intermediate=True)
        s_final = result["S_final"]
        passed = torch.isfinite(s_final).all()
        status = "[OK]" if passed else "[FAIL]"
        print(f"    {status} constant image: range [{s_final.min():.4f}, {s_final.max():.4f}]")
        if not passed:
            all_passed = False
    except Exception as e:
        print(f"    [FAIL] constant image error: {e}")
        all_passed = False

    # 6.2 渐变图像
    print("  6.2 Gradient Image Test")
    H, W = 256, 256
    y, x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    gradient = (x + y) / (2 * (H - 1))  # [0, 1]
    image_grad = gradient.unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)
    try:
        result = generate_s_final(image_grad, return_intermediate=True)
        s_final = result["S_final"]
        passed = torch.isfinite(s_final).all() and s_final.min() >= 0 and s_final.max() <= 1
        status = "[OK]" if passed else "[FAIL]"
        print(f"    {status} gradient image: range [{s_final.min():.4f}, {s_final.max():.4f}]")
        if not passed:
            all_passed = False
    except Exception as e:
        print(f"    [FAIL] gradient image error: {e}")
        all_passed = False

    # 6.3 局部高雾区域图像
    print("  6.3 Local Hazy Region Test")
    image_haze = torch.rand(1, 3, 256, 256)
    # 在中心区域增加雾密度（降低对比度）
    center_h, center_w = 64, 64
    image_haze[:, :, 100:100+center_h, 100:100+center_w] = 0.8  # 高亮度区域
    try:
        result = generate_s_final(image_haze, return_intermediate=True)
        s_final = result["S_final"]
        passed = torch.isfinite(s_final).all() and s_final.min() >= 0 and s_final.max() <= 1
        status = "[OK]" if passed else "[FAIL]"
        print(f"    {status} local haze image: range [{s_final.min():.4f}, {s_final.max():.4f}]")
        if not passed:
            all_passed = False
    except Exception as e:
        print(f"    [FAIL] local haze image error: {e}")
        all_passed = False

    if all_passed:
        print("[OK] All constructive tests passed!")
    else:
        print("[FAIL] Some constructive tests failed!")

    return all_passed


def test_constants():
    """7. Constants Test"""
    print_separator("7. Constants Test (申报书规定参数)")

    checks = [
        ("WEIGHT_DARK", WEIGHT_DARK, 0.5),
        ("WEIGHT_CONTRAST", WEIGHT_CONTRAST, 0.3),
        ("WEIGHT_COLOR", WEIGHT_COLOR, 0.2),
        ("EXPONENT_MU", EXPONENT_MU, 1.5),
    ]

    all_passed = True

    for name, value, expected in checks:
        passed = value == expected
        status = "[OK]" if passed else "[FAIL]"
        print(f"  {status} {name} = {value} (expected {expected})")
        if not passed:
            all_passed = False

    if all_passed:
        print("[OK] All constants match 申报书规定!")
    else:
        print("[FAIL] Some constants mismatch!")

    return all_passed


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("物理先验模块测试")
    print("=" * 60)

    # 检查设备
    if torch.cuda.is_available():
        print(f"\nGPU available: {torch.cuda.get_device_name(0)}")
    else:
        print("\nGPU available: No (running on CPU)")

    # 运行测试
    results = []

    results.append(("Constants", test_constants()))
    results.append(("Shape", test_shapes()))
    results.append(("Range", test_range()))
    results.append(("Finite", test_finite()))
    results.append(("Batch", test_batch()))
    results.append(("GPU", test_gpu()))
    results.append(("Constructive", test_constructive()))

    # 汇总
    print_separator("测试结果汇总")

    for name, passed in results:
        status = "[OK]" if passed else "[FAIL]"
        print(f"{status} {name}")

    all_passed = all(r[1] for r in results)

    if all_passed:
        print("\n[OK] 所有测试通过！")
        print("\n下一步:")
        print("  1. 运行可视化脚本：python scripts/visualize_physical_prior.py")
        print("  2. 进入下一阶段：Encoder + 基础模块实现")
    else:
        print("\n[FAIL] 部分测试未通过，请修复后重试")

    print("=" * 60)

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
