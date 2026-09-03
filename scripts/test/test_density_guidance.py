# -*- coding: utf-8 -*-
"""
Stage 6-3A: Density Guidance Module Test

测试 DensityGuidanceModule 的完整功能。

测试项目:
1. Shape Test (F0/F1/F2/F3)
2. Identity Initialization Test
3. Gradient Flow Test
4. Density Sensitivity Test
5. No NaN/Inf Test
6. CUDA Test
7. Parameter Statistics

使用方法:
    python scripts/test_density_guidance.py
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


def test_shape():
    """测试 1: Shape Test for F0/F1/F2/F3"""
    print_separator("Test 1: Shape Test (F0/F1/F2/F3)")

    from src.models.density_guidance import DensityGuidanceModule

    device = torch.device("cpu")
    print(f"Device: {device}")

    # 定义测试用例：(feature_channels, feature_size)
    # SimpleBackbone 实际输出：
    # F0: [B, 128, 128, 128] (H/2)
    # F1: [B, 256, 64, 64] (H/4)
    # F2: [B, 512, 32, 32] (H/8)
    # F3: [B, 1024, 16, 16] (H/16)
    test_cases = [
        ("F0", 128, 128),
        ("F1", 256, 64),
        ("F2", 512, 32),
        ("F3", 1024, 16),
    ]

    batch_size = 2
    density_size = 256  # Density map 固定为 256x256

    print("\nTest cases (density_size=256):")
    for name, feature_channels, feature_size in test_cases:
        # 创建模块
        module = DensityGuidanceModule(feature_channels=feature_channels)
        module.to(device)
        module.eval()

        # 创建输入
        visual_feature = torch.rand(batch_size, feature_channels, feature_size, feature_size)
        visual_feature = visual_feature.to(device)

        density_map = torch.rand(batch_size, 1, density_size, density_size)
        density_map = density_map.to(device)

        # Forward
        with torch.no_grad():
            guided_feature = module(visual_feature, density_map)

        # 检查输出形状
        expected_shape = visual_feature.shape
        assert guided_feature.shape == expected_shape, \
            f"{name} shape mismatch: {guided_feature.shape} != {expected_shape}"

        print(f"  {name}: PASS")
        print(f"    Input feature:  {tuple(visual_feature.shape)}")
        print(f"    Density map:    {tuple(density_map.shape)}")
        print(f"    Output feature: {tuple(guided_feature.shape)}")

    print("\n[OK] Shape test passed")
    return True


def test_identity_initialization():
    """测试 2: Identity Initialization Test"""
    print_separator("Test 2: Identity Initialization Test")

    from src.models.density_guidance import DensityGuidanceModule

    device = torch.device("cpu")
    print(f"Device: {device}")

    # 创建模块 (gamma=0 初始化)
    feature_channels = 256
    module = DensityGuidanceModule(feature_channels=feature_channels)
    module.to(device)
    module.eval()

    # 检查 gamma 初始值
    gamma_value = module.gamma.item()
    print(f"\nGamma initial value: {gamma_value:.10f}")
    assert abs(gamma_value) < 1e-6, f"Gamma should be 0 at initialization, got {gamma_value}"

    # 创建输入
    batch_size = 2
    feature_size = 64
    density_size = 256

    visual_feature = torch.rand(batch_size, feature_channels, feature_size, feature_size)
    visual_feature = visual_feature.to(device)

    density_map = torch.rand(batch_size, 1, density_size, density_size)
    density_map = density_map.to(device)

    # Forward
    with torch.no_grad():
        guided_feature = module(visual_feature, density_map)

    # 计算差异
    diff = torch.abs(guided_feature - visual_feature)
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    print(f"\nDifference (guided - feature):")
    print(f"  Max diff:  {max_diff:.10e}")
    print(f"  Mean diff: {mean_diff:.10e}")

    # 由于 gamma=0，差异应该非常小 (浮点误差级别)
    assert max_diff < 1e-5, f"Max diff should be near 0, got {max_diff}"

    print("\n[OK] Identity initialization test passed")
    return True


def test_gradient_flow():
    """测试 3: Gradient Flow Test"""
    print_separator("Test 3: Gradient Flow Test")

    from src.models.density_guidance import DensityGuidanceModule

    device = torch.device("cpu")
    print(f"Device: {device}")

    # 创建模块
    feature_channels = 256
    module = DensityGuidanceModule(feature_channels=feature_channels)
    module.to(device)
    module.train()

    # 创建输入
    batch_size = 2
    feature_size = 64
    density_size = 256

    visual_feature = torch.rand(batch_size, feature_channels, feature_size, feature_size, requires_grad=True)
    visual_feature = visual_feature.to(device)

    density_map = torch.rand(batch_size, 1, density_size, density_size, requires_grad=False)
    density_map = density_map.to(device)

    # Forward
    guided_feature = module(visual_feature, density_map)

    # 创建 loss
    loss = guided_feature.sum()
    loss.backward()

    # 检查梯度
    visual_has_grad = visual_feature.grad is not None
    module_has_grad = all(p.grad is not None for p in module.parameters())

    print(f"\nGradient status:")
    print(f"  visual_feature has grad: {visual_has_grad} (should be True)")
    print(f"  module params have grad: {module_has_grad} (should be True)")

    assert visual_has_grad, "visual_feature should have gradient"
    assert module_has_grad, "module parameters should have gradients"

    # 检查具体参数梯度
    print(f"\nParameter gradients:")
    print(f"  density_proj.grad: {module.density_proj.weight.grad is not None}")
    print(f"  visual_proj.grad:  {module.visual_proj.weight.grad is not None}")
    print(f"  gamma.grad:        {module.gamma.grad is not None}")

    print("\n[OK] Gradient flow test passed")
    return True


def test_density_sensitivity():
    """测试 4: Density Sensitivity Test"""
    print_separator("Test 4: Density Sensitivity Test")

    from src.models.density_guidance import DensityGuidanceModule

    device = torch.device("cpu")
    print(f"Device: {device}")

    # 创建模块并设置 gamma 为非零值
    feature_channels = 256
    module = DensityGuidanceModule(feature_channels=feature_channels)
    module.to(device)
    module.train()

    # 设置 gamma 为非零值 (否则即使 density 不同，output 也相同)
    with torch.no_grad():
        module.gamma.fill_(0.5)

    # 创建输入
    batch_size = 2
    feature_size = 64
    density_size = 256

    visual_feature = torch.rand(batch_size, feature_channels, feature_size, feature_size)
    visual_feature = visual_feature.to(device)

    # 两个不同的 density map
    density_zeros = torch.zeros(batch_size, 1, density_size, density_size)
    density_zeros = density_zeros.to(device)

    density_ones = torch.ones(batch_size, 1, density_size, density_size)
    density_ones = density_ones.to(device)

    # Forward
    with torch.no_grad():
        guided_zeros = module(visual_feature, density_zeros)
        guided_ones = module(visual_feature, density_ones)

    # 计算差异
    diff = torch.abs(guided_ones - guided_zeros)
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    print(f"\nDifference (guided_ones - guided_zeros):")
    print(f"  Max diff:  {max_diff:.6f}")
    print(f"  Mean diff: {mean_diff:.6f}")

    # 差异应该明显大于 0
    assert max_diff > 1e-3, f"Density should affect output, max diff={max_diff}"

    print("\n[OK] Density sensitivity test passed")
    return True


def test_no_nan_inf():
    """测试 5: No NaN/Inf Test"""
    print_separator("Test 5: No NaN/Inf Test")

    from src.models.density_guidance import DensityGuidanceModule

    device = torch.device("cpu")
    print(f"Device: {device}")

    # 创建模块
    feature_channels = 256
    module = DensityGuidanceModule(feature_channels=feature_channels)
    module.to(device)
    module.eval()

    # 定义测试用例
    test_cases = [
        ("Random feature + Random density",
         lambda: (torch.rand(2, feature_channels, 64, 64), torch.rand(2, 1, 256, 256))),
        ("Random feature + Zero density",
         lambda: (torch.rand(2, feature_channels, 64, 64), torch.zeros(2, 1, 256, 256))),
        ("Random feature + One density",
         lambda: (torch.rand(2, feature_channels, 64, 64), torch.ones(2, 1, 256, 256))),
        ("Zero feature + Random density",
         lambda: (torch.zeros(2, feature_channels, 64, 64), torch.rand(2, 1, 256, 256))),
        ("One feature + Random density",
         lambda: (torch.ones(2, feature_channels, 64, 64), torch.rand(2, 1, 256, 256))),
        ("Small feature + Random density",
         lambda: (torch.rand(2, feature_channels, 64, 64) * 0.01, torch.rand(2, 1, 256, 256))),
        ("Large feature + Random density",
         lambda: (torch.rand(2, feature_channels, 64, 64) * 100, torch.rand(2, 1, 256, 256))),
    ]

    print("\nTest cases:")
    for name, gen_fn in test_cases:
        visual_feature, density_map = gen_fn()
        visual_feature = visual_feature.to(device)
        density_map = density_map.to(device)

        with torch.no_grad():
            guided_feature = module(visual_feature, density_map)

        # 检查 NaN
        has_nan = torch.isnan(guided_feature).any().item()

        # 检查 Inf
        has_inf = torch.isinf(guided_feature).any().item()

        status = "PASS" if not (has_nan or has_inf) else "FAIL"
        print(f"  {name:40s}: NaN={str(has_nan):5s}, Inf={str(has_inf):5s} → {status}")

        assert not has_nan, f"NaN in output for {name}"
        assert not has_inf, f"Inf in output for {name}"

    print("\n[OK] No NaN/Inf test passed")
    return True


def test_cuda():
    """测试 6: CUDA Test"""
    print_separator("Test 6: CUDA Test")

    if not torch.cuda.is_available():
        print("[SKIP] CUDA not available")
        return True

    from src.models.density_guidance import DensityGuidanceModule

    device = torch.device("cuda")
    print(f"Device: {device} ({torch.cuda.get_device_name(0)})")

    # 定义测试用例
    test_cases = [
        ("F0", 128, 128),
        ("F1", 256, 64),
        ("F2", 512, 32),
        ("F3", 1024, 16),
    ]

    batch_size = 4
    density_size = 256

    print("\nTest cases (batch=4, density_size=256):")
    for name, feature_channels, feature_size in test_cases:
        # 创建模块
        module = DensityGuidanceModule(feature_channels=feature_channels)
        module.to(device)
        module.eval()

        # 创建输入
        visual_feature = torch.rand(batch_size, feature_channels, feature_size, feature_size)
        visual_feature = visual_feature.to(device)

        density_map = torch.rand(batch_size, 1, density_size, density_size)
        density_map = density_map.to(device)

        # Warmup
        for _ in range(3):
            with torch.no_grad():
                guided_feature = module(visual_feature, density_map)

        # Timing
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(10):
            with torch.no_grad():
                guided_feature = module(visual_feature, density_map)
        torch.cuda.synchronize()
        end = time.time()

        avg_latency = (end - start) / 10 * 1000  # ms

        # 检查输出
        assert guided_feature.shape == visual_feature.shape, f"Shape mismatch for {name}"
        assert guided_feature.device.type == "cuda", "Output not on CUDA"
        assert torch.isfinite(guided_feature).all(), f"Non-finite output for {name}"

        print(f"  {name}: PASS")
        print(f"    Latency: {avg_latency:.3f} ms")

    print("\n[OK] CUDA test passed")
    return True


def test_parameter_statistics():
    """测试 7: Parameter Statistics"""
    print_separator("Test 7: Parameter Statistics")

    from src.models.density_guidance import DensityGuidanceModule

    # 定义测试用例
    test_cases = [
        ("F0", 128),
        ("F1", 256),
        ("F2", 512),
        ("F3", 1024),
    ]

    print("\nParameter counts per feature scale:")
    total_params = 0

    for name, feature_channels in test_cases:
        module = DensityGuidanceModule(feature_channels=feature_channels)

        param_count = module.count_parameters()
        total_params += param_count

        # 详细统计
        density_proj_params = module.density_proj.weight.numel()
        visual_proj_params = module.visual_proj.weight.numel()
        gamma_params = module.gamma.numel()

        print(f"\n  {name} (C={feature_channels}):")
        print(f"    Total params:       {param_count:,}")
        print(f"    density_proj:       {density_proj_params:,} (1×1×1×{feature_channels})")
        print(f"    visual_proj:        {visual_proj_params:,} ({feature_channels}×{feature_channels}×1×1)")
        print(f"    gamma:              {gamma_params:,}")

    print(f"\nTotal parameters (all 4 scales): {total_params:,}")

    # 验证计算
    # density_proj: 1×C = C
    # visual_proj: C×C = C²
    # gamma: 1
    # Total per scale: C + C² + 1
    expected_total = sum(c + c*c + 1 for c in [128, 256, 512, 1024])
    print(f"Expected total:             {expected_total:,}")

    assert total_params == expected_total, f"Parameter count mismatch: {total_params} != {expected_total}"

    print("\n[OK] Parameter statistics test passed")
    return True


def test_multi_scale_integration():
    """测试 8: Multi-scale Integration Test"""
    print_separator("Test 8: Multi-scale Integration Test")

    from src.models.density_guidance import create_density_guidance_modules

    device = torch.device("cpu")
    print(f"Device: {device}")

    # 创建多尺度模块
    modules = create_density_guidance_modules()
    modules.to(device)
    modules.eval()

    print(f"\nCreated {len(modules)} guidance modules:")
    for i, module in enumerate(modules):
        print(f"  Module {i}: {module}")

    # 模拟 SimpleBackbone 输出
    batch_size = 2
    image_size = 256

    # SimpleBackbone 输出：
    # F0: [B, 128, 128, 128]
    # F1: [B, 256, 64, 64]
    # F2: [B, 512, 32, 32]
    # F3: [B, 1024, 16, 16]
    features = [
        torch.rand(batch_size, 128, image_size // 2, image_size // 2),
        torch.rand(batch_size, 256, image_size // 4, image_size // 4),
        torch.rand(batch_size, 512, image_size // 8, image_size // 8),
        torch.rand(batch_size, 1024, image_size // 16, image_size // 16),
    ]

    # Density map: [B, 1, 256, 256]
    density_map = torch.rand(batch_size, 1, image_size, image_size)

    # Forward
    with torch.no_grad():
        guided_features = []
        for i, (module, feature) in enumerate(zip(modules, features)):
            guided = module(feature, density_map)
            guided_features.append(guided)
            assert guided.shape == feature.shape, f"Module {i} shape mismatch"

    print(f"\nMulti-scale forward:")
    for i, (feat, guided) in enumerate(zip(features, guided_features)):
        print(f"  Scale {i}: {tuple(feat.shape)} → {tuple(guided.shape)}")

    print("\n[OK] Multi-scale integration test passed")
    return True


def main():
    """运行所有测试"""
    print_separator("Stage 6-3A: Density Guidance Module Test")

    tests = [
        ("Shape Test", test_shape),
        ("Identity Init", test_identity_initialization),
        ("Gradient Flow", test_gradient_flow),
        ("Density Sensitivity", test_density_sensitivity),
        ("No NaN/Inf", test_no_nan_inf),
        ("CUDA Test", test_cuda),
        ("Parameter Stats", test_parameter_statistics),
        ("Multi-scale Integration", test_multi_scale_integration),
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
        print(f"  {name:25s}: {status}")

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
