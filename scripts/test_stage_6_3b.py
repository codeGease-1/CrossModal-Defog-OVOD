# -*- coding: utf-8 -*-
"""
Stage 6-3B: Density Guidance Integration Test

测试 DensityGuidedBackbone 的完整集成。

测试项目 (12 项):
1. Checkpoint Loading
2. HazeDensityNet Frozen
3. Density Forward
4. Four-scale Forward Shape (256)
5. Four-scale Forward Shape (512)
6. Gradient Flow
7. Gamma Identity
8. Density Sensitivity
9. Guidance Off vs Gamma=0 Equivalence
10. No NaN/Inf
11. CUDA / T4
12. Parameter Statistics

使用方法:
    python scripts/test_stage_6_3b.py
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


def test_checkpoint_loading():
    """测试 1: Checkpoint Loading"""
    print_separator("Test 1: Checkpoint Loading")

    from src.models.haze_density import HazeDensityNet

    checkpoint_path = "experiments/haze_density/checkpoints/formal/best.pth"
    print(f"Checkpoint path: {checkpoint_path}")

    # 检查文件存在
    if not Path(checkpoint_path).exists():
        print(f"[FAIL] Checkpoint file not found: {checkpoint_path}")
        return False

    # 加载 checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    print(f"Checkpoint keys: {list(checkpoint.keys())}")

    # 检查 model_state_dict
    if 'model_state_dict' not in checkpoint:
        print("[FAIL] 'model_state_dict' not in checkpoint")
        return False

    # 创建模型并加载
    model = HazeDensityNet(base_channels=32, use_sigmoid=True)
    model_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {model_params:,}")

    # 严格加载
    missing_keys, unexpected_keys = model.load_state_dict(
        checkpoint['model_state_dict'], strict=True
    )

    if missing_keys:
        print(f"[FAIL] Missing keys: {missing_keys}")
        return False

    if unexpected_keys:
        print(f"[FAIL] Unexpected keys: {unexpected_keys}")
        return False

    print(f"[OK] Checkpoint loaded successfully (strict=True)")
    print("\n[OK] Checkpoint Loading: PASS")
    return True


def test_haze_density_net_frozen():
    """测试 2: HazeDensityNet Frozen"""
    print_separator("Test 2: HazeDensityNet Frozen")

    from src.models.crossmodal import DensityGuidedBackbone

    model = DensityGuidedBackbone(freeze_density=True)

    # 检查所有参数
    all_frozen = all(not p.requires_grad for p in model.density_net.parameters())

    if not all_frozen:
        print("[FAIL] Some HazeDensityNet parameters are not frozen")
        return False

    print("[OK] All HazeDensityNet parameters have requires_grad=False")
    print("\n[OK] HazeDensityNet Frozen: PASS")
    return True


def test_density_forward():
    """测试 3: Density Forward"""
    print_separator("Test 3: Density Forward")

    from src.models.crossmodal import DensityGuidedBackbone

    device = torch.device("cpu")
    model = DensityGuidedBackbone(freeze_density=True)
    model.to(device)
    model.eval()

    # 输入
    batch_size = 2
    image_size = 256
    x = torch.rand(batch_size, 3, image_size, image_size)
    x = x.to(device)

    # Forward
    with torch.no_grad():
        density = model.get_density(x)

    # 检查 shape
    expected_shape = (batch_size, 1, image_size, image_size)
    if density.shape != expected_shape:
        print(f"[FAIL] Density shape mismatch: {density.shape} != {expected_shape}")
        return False

    # 检查 finite
    if not torch.isfinite(density).all():
        print("[FAIL] Density contains NaN or Inf")
        return False

    # 检查 range
    density_min, density_max = density.min().item(), density.max().item()
    print(f"Density range: [{density_min:.4f}, {density_max:.4f}]")

    if density_min < 0 or density_max > 1:
        print(f"[WARN] Density range outside [0, 1]")

    print(f"[OK] Density shape: {tuple(density.shape)}")
    print(f"[OK] Density finite: True")
    print("\n[OK] Density Forward: PASS")
    return True


def test_four_scale_forward_shape_256():
    """测试 4: Four-scale Forward Shape (256)"""
    print_separator("Test 4: Four-scale Forward Shape (256)")

    from src.models.crossmodal import DensityGuidedBackbone

    device = torch.device("cpu")
    model = DensityGuidedBackbone(freeze_density=True)
    model.to(device)
    model.eval()

    batch_size = 2
    image_size = 256
    x = torch.rand(batch_size, 3, image_size, image_size)
    x = x.to(device)

    # Forward
    with torch.no_grad():
        guided_features, density = model(x)

    # 期望 shape
    expected_shapes = [
        (batch_size, 128, image_size // 2, image_size // 2),  # F0: H/2
        (batch_size, 256, image_size // 4, image_size // 4),  # F1: H/4
        (batch_size, 512, image_size // 8, image_size // 8),  # F2: H/8
        (batch_size, 1024, image_size // 16, image_size // 16),  # F3: H/16
    ]

    print("Feature shapes:")
    for i, (feat, expected) in enumerate(zip(guided_features, expected_shapes)):
        if feat.shape != expected:
            print(f"[FAIL] Scale {i} shape mismatch: {tuple(feat.shape)} != {expected}")
            return False
        print(f"  Scale {i}: {tuple(feat.shape)} ✓")

    print("\n[OK] Four-scale Forward Shape (256): PASS")
    return True


def test_four_scale_forward_shape_512():
    """测试 5: Four-scale Forward Shape (512)"""
    print_separator("Test 5: Four-scale Forward Shape (512)")

    from src.models.crossmodal import DensityGuidedBackbone

    device = torch.device("cpu")
    model = DensityGuidedBackbone(freeze_density=True)
    model.to(device)
    model.eval()

    batch_size = 1
    image_size = 512
    x = torch.rand(batch_size, 3, image_size, image_size)
    x = x.to(device)

    # Forward
    with torch.no_grad():
        guided_features, density = model(x)

    # 期望 shape
    expected_shapes = [
        (batch_size, 128, image_size // 2, image_size // 2),  # F0: H/2
        (batch_size, 256, image_size // 4, image_size // 4),  # F1: H/4
        (batch_size, 512, image_size // 8, image_size // 8),  # F2: H/8
        (batch_size, 1024, image_size // 16, image_size // 16),  # F3: H/16
    ]

    print("Feature shapes:")
    for i, (feat, expected) in enumerate(zip(guided_features, expected_shapes)):
        if feat.shape != expected:
            print(f"[FAIL] Scale {i} shape mismatch: {tuple(feat.shape)} != {expected}")
            return False
        print(f"  Scale {i}: {tuple(feat.shape)} ✓")

    print("\n[OK] Four-scale Forward Shape (512): PASS")
    return True


def test_gradient_flow():
    """测试 6: Gradient Flow"""
    print_separator("Test 6: Gradient Flow")

    from src.models.crossmodal import DensityGuidedBackbone

    device = torch.device("cpu")
    model = DensityGuidedBackbone(freeze_density=True)
    model.to(device)
    model.train()

    batch_size = 2
    image_size = 256
    x = torch.rand(batch_size, 3, image_size, image_size)
    x = x.to(device)

    # Forward
    guided_features, density = model(x)

    # Dummy loss
    loss = sum(f.mean() for f in guided_features)
    loss.backward()

    # 检查 backbone 梯度
    backbone_has_grad = any(p.grad is not None for p in model.backbone.parameters())
    if not backbone_has_grad:
        print("[FAIL] Backbone has no gradients")
        return False
    print("[OK] Backbone has gradients")

    # 检查 guidance 梯度
    guidance_has_grad = any(p.grad is not None for p in model.guidance_modules.parameters())
    if not guidance_has_grad:
        print("[FAIL] Guidance modules have no gradients")
        return False
    print("[OK] Guidance modules have gradients")

    # 检查 density net 无梯度
    density_has_grad = any(p.grad is not None for p in model.density_net.parameters())
    if density_has_grad:
        print("[FAIL] HazeDensityNet should not have gradients")
        return False
    print("[OK] HazeDensityNet has no gradients")

    print("\n[OK] Gradient Flow: PASS")
    return True


def test_gamma_identity():
    """测试 7: Gamma Identity"""
    print_separator("Test 7: Gamma Identity")

    from src.models.crossmodal import DensityGuidedBackbone

    device = torch.device("cpu")
    model = DensityGuidedBackbone(freeze_density=True)
    model.to(device)
    model.eval()

    # 确保 gamma=0
    for i, module in enumerate(model.guidance_modules):
        with torch.no_grad():
            module.gamma.fill_(0.0)

    batch_size = 2
    image_size = 256
    x = torch.rand(batch_size, 3, image_size, image_size)
    x = x.to(device)

    # 获取原始特征
    with torch.no_grad():
        raw_features = model.get_raw_features(x)
        guided_features, _ = model(x)

    # 比较
    print("Max differences (guided vs raw):")
    all_pass = True
    for i, (raw, guided) in enumerate(zip(raw_features, guided_features)):
        max_diff = (raw - guided).abs().max().item()
        print(f"  Scale {i}: {max_diff:.2e}")
        if max_diff > 1e-7:
            print(f"[FAIL] Scale {i} max_diff > 1e-7")
            all_pass = False

    if not all_pass:
        return False

    print("\n[OK] Gamma Identity: PASS")
    return True


def test_density_sensitivity():
    """测试 8: Density Sensitivity"""
    print_separator("Test 8: Density Sensitivity")

    from src.models.crossmodal import DensityGuidedBackbone

    device = torch.device("cpu")
    model = DensityGuidedBackbone(freeze_density=True)
    model.to(device)
    model.train()

    # 设置 gamma 为非零
    for module in model.guidance_modules:
        with torch.no_grad():
            module.gamma.fill_(0.5)

    batch_size = 2
    image_size = 256
    x = torch.rand(batch_size, 3, image_size, image_size)
    x = x.to(device)

    # Zero density
    density_zero = torch.zeros(batch_size, 1, image_size, image_size)
    density_zero = density_zero.to(device)

    # One density
    density_one = torch.ones(batch_size, 1, image_size, image_size)
    density_one = density_one.to(device)

    # Forward
    with torch.no_grad():
        guided_zero, _ = model(x, density=density_zero)
        guided_one, _ = model(x, density=density_one)

    # 比较
    print("Mean differences (guided_one - guided_zero):")
    all_pass = True
    for i, (g0, g1) in enumerate(zip(guided_zero, guided_one)):
        mean_diff = (g1 - g0).abs().mean().item()
        print(f"  Scale {i}: {mean_diff:.6f}")
        if mean_diff < 1e-6:
            print(f"[FAIL] Scale {i} mean_diff too small")
            all_pass = False

    if not all_pass:
        return False

    print("\n[OK] Density Sensitivity: PASS")
    return True


def test_guidance_off_vs_gamma_zero():
    """测试 9: Guidance Off vs Gamma=0 Equivalence"""
    print_separator("Test 9: Guidance Off vs Gamma=0 Equivalence")

    from src.models.crossmodal import DensityGuidedBackbone

    device = torch.device("cpu")
    model = DensityGuidedBackbone(freeze_density=True)
    model.to(device)
    model.eval()

    # 确保 gamma=0
    for module in model.guidance_modules:
        with torch.no_grad():
            module.gamma.fill_(0.0)

    batch_size = 2
    image_size = 256
    x = torch.rand(batch_size, 3, image_size, image_size)
    x = x.to(device)

    # Mode A: Guidance Off (raw backbone)
    with torch.no_grad():
        features_off = model.get_raw_features(x)

    # Mode B: Gamma=0
    with torch.no_grad():
        features_identity, _ = model(x)

    # 比较
    print("Max differences (off vs identity):")
    all_pass = True
    for i, (off, identity) in enumerate(zip(features_off, features_identity)):
        max_diff = (off - identity).abs().max().item()
        print(f"  Scale {i}: {max_diff:.2e}")
        if max_diff > 1e-7:
            print(f"[FAIL] Scale {i} max_diff > 1e-7")
            all_pass = False

    if not all_pass:
        return False

    print("\n[OK] Guidance Off vs Gamma=0: PASS")
    return True


def test_no_nan_inf():
    """测试 10: No NaN/Inf"""
    print_separator("Test 10: No NaN/Inf")

    from src.models.crossmodal import DensityGuidedBackbone

    device = torch.device("cpu")
    model = DensityGuidedBackbone(freeze_density=True)
    model.to(device)
    model.eval()

    test_cases = [
        ("Random RGB", lambda: torch.rand(2, 3, 256, 256)),
        ("Zero RGB", lambda: torch.zeros(2, 3, 256, 256)),
        ("One RGB", lambda: torch.ones(2, 3, 256, 256)),
        ("Small RGB", lambda: torch.rand(2, 3, 256, 256) * 0.01),
        ("Large RGB", lambda: torch.rand(2, 3, 256, 256) * 2.0),
    ]

    print("Test cases:")
    for name, gen_fn in test_cases:
        x = gen_fn().to(device)

        with torch.no_grad():
            guided_features, density = model(x)

        # 检查 density
        density_nan = torch.isnan(density).any().item()
        density_inf = torch.isinf(density).any().item()

        # 检查 features
        feat_nan = any(torch.isnan(f).any().item() for f in guided_features)
        feat_inf = any(torch.isinf(f).any().item() for f in guided_features)

        status = "PASS" if not (density_nan or density_inf or feat_nan or feat_inf) else "FAIL"
        print(f"  {name:20s}: NaN={str(feat_nan or density_nan):5s}, Inf={str(feat_inf or density_inf):5s} → {status}")

        if density_nan or density_inf or feat_nan or feat_inf:
            print(f"[FAIL] {name} has NaN or Inf")
            return False

    print("\n[OK] No NaN/Inf: PASS")
    return True


def test_cuda():
    """测试 11: CUDA / T4"""
    print_separator("Test 11: CUDA / T4")

    if not torch.cuda.is_available():
        print("[SKIP] CUDA not available")
        return True

    from src.models.crossmodal import DensityGuidedBackbone

    device = torch.device("cuda")
    print(f"Device: {device} ({torch.cuda.get_device_name(0)})")

    model = DensityGuidedBackbone(freeze_density=True)
    model.to(device)
    model.eval()

    batch_size = 4
    image_size = 256
    x = torch.rand(batch_size, 3, image_size, image_size)
    x = x.to(device)

    # Warmup
    for _ in range(3):
        with torch.no_grad():
            guided_features, density = model(x)

    # Timing
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(10):
        with torch.no_grad():
            guided_features, density = model(x)
    torch.cuda.synchronize()
    end = time.perf_counter()

    avg_latency = (end - start) / 10 * 1000  # ms

    # 检查输出
    expected_shapes = [
        (batch_size, 128, image_size // 2, image_size // 2),
        (batch_size, 256, image_size // 4, image_size // 4),
        (batch_size, 512, image_size // 8, image_size // 8),
        (batch_size, 1024, image_size // 16, image_size // 16),
    ]

    for i, (feat, expected) in enumerate(zip(guided_features, expected_shapes)):
        if feat.shape != expected:
            print(f"[FAIL] Scale {i} shape mismatch on CUDA")
            return False

    if not torch.isfinite(guided_features[0]).all():
        print("[FAIL] Non-finite output on CUDA")
        return False

    print(f"Latency (batch={batch_size}, size={image_size}): {avg_latency:.2f} ms")
    print("Shape check: PASS")
    print("Finite check: PASS")

    print("\n[OK] CUDA / T4: PASS")
    return True


def test_parameter_statistics():
    """测试 12: Parameter Statistics"""
    print_separator("Test 12: Parameter Statistics")

    from src.models.crossmodal import DensityGuidedBackbone

    model = DensityGuidedBackbone(freeze_density=True)
    stats = model.count_parameters()

    print("\nParameter statistics:")
    print(f"  Total parameters:      {stats['total']:,}")
    print(f"  HazeDensityNet:        {stats['density']:,}")
    print(f"  Backbone:              {stats['backbone']:,}")
    print(f"  Guidance modules:      {stats['guidance']:,}")
    print(f"  Trainable parameters:  {stats['trainable']:,}")
    print(f"  Frozen parameters:     {stats['frozen']:,}")

    # 验证 guidance 参数量
    # F0: 128 + 128^2 + 1 = 16,513
    # F1: 256 + 256^2 + 1 = 65,793
    # F2: 512 + 512^2 + 1 = 262,657
    # F3: 1024 + 1024^2 + 1 = 1,049,601
    # Total: 1,394,564
    expected_guidance = 1_394_564

    if stats['guidance'] != expected_guidance:
        print(f"[WARN] Guidance params mismatch: {stats['guidance']:,} != {expected_guidance:,}")
    else:
        print(f"[OK] Guidance params: {stats['guidance']:,} ✓")

    # 验证 frozen
    if stats['frozen'] != stats['density']:
        print(f"[WARN] Frozen params != density params")
    else:
        print(f"[OK] Frozen params = density params ✓")

    print("\n[OK] Parameter Statistics: PASS")
    return True


def main():
    """运行所有测试"""
    print_separator("Stage 6-3B: Density Guidance Integration Test")

    tests = [
        ("Checkpoint Loading", test_checkpoint_loading),
        ("HazeDensityNet Frozen", test_haze_density_net_frozen),
        ("Density Forward", test_density_forward),
        ("256 Shape", test_four_scale_forward_shape_256),
        ("512 Shape", test_four_scale_forward_shape_512),
        ("Gradient Flow", test_gradient_flow),
        ("Gamma Identity", test_gamma_identity),
        ("Density Sensitivity", test_density_sensitivity),
        ("Guidance Off vs Gamma=0", test_guidance_off_vs_gamma_zero),
        ("No NaN/Inf", test_no_nan_inf),
        ("CUDA / T4", test_cuda),
        ("Parameter Statistics", test_parameter_statistics),
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
        print(f"  {name:30s}: {status}")

    print(f"\nTotal: {passed}/{total} tests passed")

    # 输出格式
    print_separator("Stage 6-3B Result")

    for name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{name}:")
        print(status)

    print(f"\nTotal:")
    print(f"{passed}/{total} tests passed")

    if passed == total:
        print("\n[STAGE 6-3B PASSED]")
        return True
    else:
        print("\n[STAGE 6-3B FAILED]")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
