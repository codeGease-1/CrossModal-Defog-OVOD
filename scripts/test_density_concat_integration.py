#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stage 6-2: Density Concatenation Integration Test

测试 Density Concatenation Baseline 的完整 forward 流程。

测试项目:
1. Forward shape test
2. CUDA test
3. Gradient flow test
4. Frozen density network test
5. No NaN/Inf test

使用方法:
    python scripts/test_density_concat_integration.py
"""

import sys
from pathlib import Path

# 设置路径
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import torch
import time


def print_separator(title: str):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def test_forward_shape():
    """测试 1: Forward shape test"""
    print_separator("Test 1: Forward Shape Test")

    from src.models.crossmodal import DensityConcatModel

    device = torch.device("cpu")
    print(f"Device: {device}")

    # 创建模型
    print("\nCreating model...")
    model = DensityConcatModel(
        density_checkpoint="experiments/haze_density/checkpoints/formal/best.pth",
        freeze_density=True,
    )
    model.to(device)
    model.eval()

    # 测试不同 batch size 和 image size
    test_cases = [
        (1, 256),
        (2, 256),
        (4, 256),
        (1, 512),
    ]

    print("\nTest cases:")
    for batch_size, image_size in test_cases:
        hazy_image = torch.rand(batch_size, 3, image_size, image_size)
        hazy_image = hazy_image.to(device)

        with torch.no_grad():
            output = model(hazy_image, return_density=True)

        # 检查输出形状
        density_map = output['density_map']
        concat_input = output['concat_input']
        features = output['features']

        expected_density_shape = (batch_size, 1, image_size, image_size)
        expected_concat_shape = (batch_size, 4, image_size, image_size)
        expected_feature_shapes = [
            (batch_size, 128, image_size // 4, image_size // 4),
            (batch_size, 256, image_size // 8, image_size // 8),
            (batch_size, 512, image_size // 16, image_size // 16),
            (batch_size, 1024, image_size // 32, image_size // 32),
        ]

        assert density_map.shape == expected_density_shape, \
            f"Density shape mismatch: {density_map.shape} != {expected_density_shape}"
        assert concat_input.shape == expected_concat_shape, \
            f"Concat shape mismatch: {concat_input.shape} != {expected_concat_shape}"

        for i, (feat, expected) in enumerate(zip(features, expected_feature_shapes)):
            assert feat.shape == expected, \
                f"Feature {i} shape mismatch: {feat.shape} != {expected}"

        print(f"  Batch={batch_size}, Size={image_size}: PASS")
        print(f"    RGB:        {tuple(hazy_image.shape)}")
        print(f"    Density:    {tuple(density_map.shape)}")
        print(f"    Concat:     {tuple(concat_input.shape)}")
        print(f"    Features:   {[tuple(f.shape) for f in features]}")

    print("\n[OK] Forward shape test passed")
    return True


def test_cuda():
    """测试 2: CUDA test"""
    print_separator("Test 2: CUDA Test")

    if not torch.cuda.is_available():
        print("[SKIP] CUDA not available")
        return True

    from src.models.crossmodal import DensityConcatModel

    device = torch.device("cuda")
    print(f"Device: {device} ({torch.cuda.get_device_name(0)})")

    # 创建模型
    print("\nCreating model...")
    model = DensityConcatModel(
        density_checkpoint="experiments/haze_density/checkpoints/formal/best.pth",
        freeze_density=True,
    )
    model.to(device)
    model.eval()

    # 测试 forward
    batch_size = 4
    image_size = 256

    hazy_image = torch.rand(batch_size, 3, image_size, image_size)
    hazy_image = hazy_image.to(device)

    # Warmup
    for _ in range(3):
        with torch.no_grad():
            output = model(hazy_image, return_density=True)

    # Timing
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(10):
        with torch.no_grad():
            output = model(hazy_image, return_density=True)
    torch.cuda.synchronize()
    end = time.time()

    avg_latency = (end - start) / 10 * 1000  # ms

    print(f"\nTiming (10 iterations, batch={batch_size}, size={image_size}):")
    print(f"  Average latency: {avg_latency:.2f} ms")
    print(f"  Throughput: {batch_size / (avg_latency / 1000):.2f} images/sec")

    # 检查输出
    density_map = output['density_map']
    assert density_map.device.type == "cuda", "Output not on CUDA"

    print("\n[OK] CUDA test passed")
    return True


def test_gradient_flow():
    """测试 3: Gradient flow test"""
    print_separator("Test 3: Gradient Flow Test")

    from src.models.crossmodal import DensityConcatModel

    device = torch.device("cpu")
    print(f"Device: {device}")

    # 创建模型 (density frozen, backbone trainable)
    print("\nCreating model (freeze_density=True)...")
    model = DensityConcatModel(
        density_checkpoint="experiments/haze_density/checkpoints/formal/best.pth",
        freeze_density=True,
    )
    model.to(device)
    model.train()

    # 检查参数状态
    density_frozen = all(not p.requires_grad for p in model.density_net.parameters())
    backbone_trainable = all(p.requires_grad for p in model.backbone.parameters())

    print(f"\nParameter status:")
    print(f"  Density net frozen:   {density_frozen}")
    print(f"  Backbone trainable:   {backbone_trainable}")

    assert density_frozen, "Density net should be frozen"
    assert backbone_trainable, "Backbone should be trainable"

    # 测试梯度流
    batch_size = 2
    image_size = 256

    hazy_image = torch.rand(batch_size, 3, image_size, image_size, requires_grad=True)
    hazy_image = hazy_image.to(device)

    output = model(hazy_image, return_density=True)
    features = output['features']

    # 创建一个简单的 loss (sum of last feature)
    loss = features[-1].sum()
    loss.backward()

    # 检查梯度
    density_has_grad = any(p.grad is not None for p in model.density_net.parameters())
    backbone_has_grad = any(p.grad is not None for p in model.backbone.parameters())

    print(f"\nGradient status:")
    print(f"  Density net has grad:   {density_has_grad} (should be False)")
    print(f"  Backbone has grad:      {backbone_has_grad} (should be True)")

    assert not density_has_grad, "Density net should not have gradients"
    assert backbone_has_grad, "Backbone should have gradients"

    print("\n[OK] Gradient flow test passed")
    return True


def test_frozen_density_network():
    """测试 4: Frozen density network test"""
    print_separator("Test 4: Frozen Density Network Test")

    from src.models.crossmodal import DensityConcatModel

    device = torch.device("cpu")
    print(f"Device: {device}")

    # 创建模型
    print("\nCreating model (freeze_density=True)...")
    model = DensityConcatModel(
        density_checkpoint="experiments/haze_density/checkpoints/formal/best.pth",
        freeze_density=True,
    )
    model.to(device)
    model.eval()

    # 测试多次 forward，密度图应该一致
    batch_size = 2
    image_size = 256

    hazy_image = torch.rand(batch_size, 3, image_size, image_size)
    hazy_image = hazy_image.to(device)

    with torch.no_grad():
        output1 = model(hazy_image, return_density=True)
        output2 = model(hazy_image, return_density=True)

    density1 = output1['density_map']
    density2 = output2['density_map']

    # 检查一致性
    diff = (density1 - density2).abs().max().item()
    print(f"\nDensity consistency test:")
    print(f"  Max difference between two forwards: {diff:.10f}")

    assert diff < 1e-6, "Density map should be consistent"

    # 测试切换 freeze 状态
    print("\nTesting freeze toggle...")
    model.set_freeze_density(False)
    model.train()

    density_unfrozen = all(p.requires_grad for p in model.density_net.parameters())
    print(f"  Density net trainable: {density_unfrozen}")

    model.set_freeze_density(True)
    model.eval()

    density_frozen = all(not p.requires_grad for p in model.density_net.parameters())
    print(f"  Density net frozen: {density_frozen}")

    assert density_frozen, "Density net should be frozen after toggle"

    print("\n[OK] Frozen density network test passed")
    return True


def test_no_nan_inf():
    """测试 5: No NaN/Inf test"""
    print_separator("Test 5: No NaN/Inf Test")

    from src.models.crossmodal import DensityConcatModel

    device = torch.device("cpu")
    print(f"Device: {device}")

    # 创建模型
    print("\nCreating model...")
    model = DensityConcatModel(
        density_checkpoint="experiments/haze_density/checkpoints/formal/best.pth",
        freeze_density=True,
    )
    model.to(device)
    model.eval()

    # 测试不同输入
    test_cases = [
        ("Random [0,1]", lambda: torch.rand(2, 3, 256, 256)),
        ("Random [0,0.5]", lambda: torch.rand(2, 3, 256, 256) * 0.5),
        ("Random [0.5,1]", lambda: torch.rand(2, 3, 256, 256) * 0.5 + 0.5),
        ("Zeros", lambda: torch.zeros(2, 3, 256, 256)),
        ("Ones", lambda: torch.ones(2, 3, 256, 256)),
    ]

    print("\nTest cases:")
    for name, gen_fn in test_cases:
        hazy_image = gen_fn().to(device)

        with torch.no_grad():
            output = model(hazy_image, return_density=True)

        density_map = output['density_map']
        concat_input = output['concat_input']
        features = output['features']

        # 检查 NaN
        has_nan_density = torch.isnan(density_map).any().item()
        has_nan_concat = torch.isnan(concat_input).any().item()
        has_nan_features = any(torch.isnan(f).any().item() for f in features)

        # 检查 Inf
        has_inf_density = torch.isinf(density_map).any().item()
        has_inf_concat = torch.isinf(concat_input).any().item()
        has_inf_features = any(torch.isinf(f).any().item() for f in features)

        nan_status = "PASS" if not (has_nan_density or has_nan_concat or has_nan_features) else "FAIL"
        inf_status = "PASS" if not (has_inf_density or has_inf_concat or has_inf_features) else "FAIL"

        print(f"  {name:20s}: NaN={nan_status}, Inf={inf_status}")

        assert not has_nan_density, f"NaN in density_map for {name}"
        assert not has_nan_concat, f"NaN in concat_input for {name}"
        assert not has_nan_features, f"NaN in features for {name}"
        assert not has_inf_density, f"Inf in density_map for {name}"
        assert not has_inf_concat, f"Inf in concat_input for {name}"
        assert not has_inf_features, f"Inf in features for {name}"

    print("\n[OK] No NaN/Inf test passed")
    return True


def test_parameter_stats():
    """测试 6: Parameter statistics"""
    print_separator("Test 6: Parameter Statistics")

    from src.models.crossmodal import DensityConcatModel
    from src.models.backbone import SimpleBackbone

    # 创建模型
    print("\nCreating model...")
    model = DensityConcatModel(
        density_checkpoint="experiments/haze_density/checkpoints/formal/best.pth",
        freeze_density=True,
    )

    stats = model.count_parameters()

    print(f"\nParameter statistics:")
    print(f"  Total parameters:      {stats['total']:,}")
    print(f"  HazeDensityNet:        {stats['density']:,}")
    print(f"  Backbone:              {stats['backbone']:,}")
    print(f"  Trainable parameters:  {stats['trainable']:,}")
    print(f"  Frozen parameters:     {stats['total'] - stats['trainable']:,}")

    # 计算增加比例
    backbone_only = SimpleBackbone(input_channels=3).count_parameters()
    increase = stats['backbone'] - backbone_only
    increase_ratio = increase / backbone_only * 100

    print(f"\nBackbone parameter increase (vs 3-channel):")
    print(f"  3-channel backbone:    {backbone_only:,}")
    print(f"  4-channel backbone:    {stats['backbone']:,}")
    print(f"  Increase:              {increase:,} ({increase_ratio:.1f}%)")

    print("\n[OK] Parameter statistics test passed")
    return True


def main():
    """运行所有测试"""
    print_separator("Stage 6-2: Density Concatenation Integration Test")

    tests = [
        ("Forward Shape", test_forward_shape),
        ("CUDA", test_cuda),
        ("Gradient Flow", test_gradient_flow),
        ("Frozen Density", test_frozen_density_network),
        ("No NaN/Inf", test_no_nan_inf),
        ("Parameter Stats", test_parameter_stats),
    ]

    results = []
    for name, test_fn in tests:
        try:
            result = test_fn()
            results.append((name, result))
        except Exception as e:
            print(f"\n[FAIL] {name} test failed: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # 总结
    print_separator("Test Summary")

    passed = sum(1 for _, r in results if r)
    total = len(results)

    for name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"  {name:20s}: {status}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n[OK] All tests passed!")
        return True
    else:
        print("\n[FAIL] Some tests failed")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
