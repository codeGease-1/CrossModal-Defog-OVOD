#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GPU 调试脚本 - 直接测试 GPU 可用性
"""

import sys
from pathlib import Path

# 设置路径
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import torch

print("=" * 60)
print("GPU 调试")
print("=" * 60)

# 1. 检查 CUDA 可用性
print(f"\n1. torch.cuda.is_available(): {torch.cuda.is_available()}")
print(f"   torch.cuda.is_initialized(): {torch.cuda.is_initialized()}")

if torch.cuda.is_available():
    print(f"   CUDA device count: {torch.cuda.device_count()}")
    print(f"   Current device: {torch.cuda.current_device()}")
    print(f"   GPU name: {torch.cuda.get_device_name(0)}")
    print(f"   CUDA capability: {torch.cuda.get_device_capability(0)}")

# 2. 测试简单 tensor 操作
print("\n2. 测试简单 tensor 操作")
try:
    device = torch.device("cuda")
    x = torch.rand(2, 3, 32, 32, device=device)
    y = x + 1
    print(f"   [OK] tensor 操作成功：{x.device} -> {y.device}")
except Exception as e:
    print(f"   [FAIL] tensor 操作失败：{e}")

# 3. 测试简单卷积
print("\n3. 测试简单卷积")
try:
    device = torch.device("cuda")
    conv = torch.nn.Conv2d(3, 16, 3, padding=1).to(device)
    x = torch.rand(2, 3, 32, 32, device=device)
    y = conv(x)
    print(f"   [OK] 卷积操作成功：{x.shape} -> {y.shape}")
except Exception as e:
    print(f"   [FAIL] 卷积操作失败：{e}")

# 4. 测试 InstanceNorm
print("\n4. 测试 InstanceNorm")
try:
    device = torch.device("cuda")
    norm = torch.nn.InstanceNorm2d(16).to(device)
    x = torch.rand(2, 16, 32, 32, device=device)
    y = norm(x)
    print(f"   [OK] InstanceNorm 操作成功：{x.shape} -> {y.shape}")
except Exception as e:
    print(f"   [FAIL] InstanceNorm 操作失败：{e}")

# 5. 测试 ConvBlock
print("\n5. 测试 ConvBlock")
try:
    from src.models.haze_density.basic_blocks import ConvBlock
    device = torch.device("cuda")
    block = ConvBlock(3, 16, kernel_size=3).to(device)
    x = torch.rand(2, 3, 32, 32, device=device)
    y = block(x)
    print(f"   [OK] ConvBlock 成功：{x.shape} -> {y.shape}")
except Exception as e:
    print(f"   [FAIL] ConvBlock 失败：{e}")
    import traceback
    traceback.print_exc()

# 6. 测试 Encoder
print("\n6. 测试 Encoder")
try:
    from src.models.haze_density.encoder import Encoder
    device = torch.device("cuda")
    encoder = Encoder(base_channels=32).to(device)
    x = torch.rand(2, 3, 128, 128, device=device)
    y = encoder(x)
    print(f"   [OK] Encoder 成功：{x.shape} -> {y.shape}")
except Exception as e:
    print(f"   [FAIL] Encoder 失败：{e}")
    import traceback
    traceback.print_exc()

# 7. 测试 ResidualBlock
print("\n7. 测试 ResidualBlock")
try:
    from src.models.haze_density.residual_blocks import ResidualBlock
    device = torch.device("cuda")
    rb = ResidualBlock(channels=64).to(device)
    x = torch.rand(2, 64, 64, 64, device=device)
    y = rb(x)
    print(f"   [OK] ResidualBlock 成功：{x.shape} -> {y.shape}")
except Exception as e:
    print(f"   [FAIL] ResidualBlock 失败：{e}")
    import traceback
    traceback.print_exc()

# 8. 测试 DilatedResidualBlock
print("\n8. 测试 DilatedResidualBlock")
try:
    from src.models.haze_density.residual_blocks import DilatedResidualBlock
    device = torch.device("cuda")
    sdrb = DilatedResidualBlock(channels=64, dilation=2).to(device)
    x = torch.rand(2, 64, 64, 64, device=device)
    y = sdrb(x)
    print(f"   [OK] DilatedResidualBlock 成功：{x.shape} -> {y.shape}")
except Exception as e:
    print(f"   [FAIL] DilatedResidualBlock 失败：{e}")
    import traceback
    traceback.print_exc()

# 9. 测试 ECA
print("\n9. 测试 ECA")
try:
    from src.models.haze_density.eca import ECA
    device = torch.device("cuda")
    eca = ECA(channels=64).to(device)
    x = torch.rand(2, 64, 64, 64, device=device)
    y = eca(x)
    print(f"   [OK] ECA 成功：{x.shape} -> {y.shape}")
except Exception as e:
    print(f"   [FAIL] ECA 失败：{e}")
    import traceback
    traceback.print_exc()

# 10. 测试 MultiScaleBranch
print("\n10. 测试 MultiScaleBranch")
try:
    from src.models.haze_density.multiscale import MultiScaleBranch
    device = torch.device("cuda")
    branch = MultiScaleBranch(channels=64, dilation=2).to(device)
    x = torch.rand(2, 64, 64, 64, device=device)
    y = branch(x)
    print(f"   [OK] MultiScaleBranch 成功：{x.shape} -> {y.shape}")
except Exception as e:
    print(f"   [FAIL] MultiScaleBranch 失败：{e}")
    import traceback
    traceback.print_exc()

# 11. 测试 ParallelMultiScaleFeatureExtractor
print("\n11. 测试 ParallelMultiScaleFeatureExtractor")
try:
    from src.models.haze_density.multiscale import ParallelMultiScaleFeatureExtractor
    device = torch.device("cuda")
    multiscale = ParallelMultiScaleFeatureExtractor(channels=64).to(device)
    x = torch.rand(2, 64, 64, 64, device=device)
    y = multiscale(x)
    print(f"   [OK] ParallelMultiScaleFeatureExtractor 成功：{x.shape} -> {y.shape}")
except Exception as e:
    print(f"   [FAIL] ParallelMultiScaleFeatureExtractor 失败：{e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("调试完成")
print("=" * 60)
