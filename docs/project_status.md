# 项目状态跟踪

**项目名称**: CrossModal-Defog-OVOD - 跨模态语义恢复引导的含雾低质遥感影像开放词汇目标检测  
**当前模块**: 雾密度感知网络 (Haze Density Perception Network)  
**最后更新**: 2026-08-28

---

## 一、开发模式

| 环境 | 职责 |
|------|------|
| **本地 (Windows)** | 代码开发、静态检查、Git 管理 |
| **Google Colab (T4 GPU)** | PyTorch 运行、模型训练、实验验证 |

---

## 二、阶段进度

### Phase 0: 工程初始化 [OK] 已完成

| 任务 | 状态 | 说明 |
|------|------|------|
| 项目目录结构 | [OK] | 基础目录已创建 |
| `requirements.txt` | [OK] | 依赖列表已定义 |
| `configs/haze_density.yaml` | [OK] | 配置文件已创建 |
| `docs/colab.md` | [OK] | Colab 使用说明 |
| `scripts/setup_colab.py` | [OK] | Colab 环境设置 |
| `scripts/static_check.py` | [OK] | 本地静态检查 |

### Phase 1: 物理先验模块 [OK] 已完成

| 任务 | 状态 | 说明 |
|------|------|------|
| `physical_prior.py` | [OK] | Dark Channel / Local Contrast / Color Shift |
| `guided_filter.py` | [OK] | 引导滤波 |
| `scripts/test_physical_prior.py` | [OK] | 测试脚本 |
| `scripts/visualize_physical_prior.py` | [OK] | 可视化脚本 |

### Phase 2: 核心特征提取模块 [OK] 已完成（待 Colab 验证）

| 任务 | 状态 | 说明 |
|------|------|------|
| `basic_blocks.py` | [OK] | ConvBlock / DownsampleBlock |
| `encoder.py` | [OK] | Encoder 下采样 |
| `residual_blocks.py` | [OK] | RB / SDRB (dilation=2,3,4) |
| `eca.py` | [OK] | ECA 通道注意力 |
| `multiscale.py` | [OK] | 多尺度分支（3 路并行） |
| `scripts/test_core_modules.py` | [OK] | 测试脚本 |
| Shape Test | [WAIT] | 【在 Colab 执行】 |
| GPU Test | [WAIT] | 【在 Colab 执行】 |
| Backward Test | [WAIT] | 【在 Colab 执行】 |

### Phase 3: Decoder + 完整模型 [WAIT] 待开始

| 任务 | 状态 | 说明 |
|------|------|------|
| `decoder.py` | [WAIT] | Decoder 上采样 |
| `haze_density_net.py` | [WAIT] | 完整模型组装 |

### Phase 4: 训练框架 [WAIT] 待开始

| 任务 | 状态 | 说明 |
|------|------|------|
| `src/data/haze_dataset.py` | [WAIT] | Dataset 类 |
| `src/losses.py` | [WAIT] | MSE Loss |
| `src/train.py` | [WAIT] | 训练循环 |

### Phase 5: Colab Smoke Test [WAIT] 待开始

| 任务 | 状态 | 说明 |
|------|------|------|
| Colab 环境配置 | [WAIT] | 安装依赖、检查 GPU |
| 完整流程测试 | [WAIT] | forward + backward |

---

## 三、技术路线确认

### 申报书规定参数（不可修改）

| 参数 | 值 | 来源 | 实现位置 |
|------|-----|------|----------|
| `weight_dark` | 0.5 | 申报书 3.2.1 | `physical_prior.py` |
| `weight_contrast` | 0.3 | 申报书 3.2.1 | `physical_prior.py` |
| `weight_color` | 0.2 | 申报书 3.2.1 | `physical_prior.py` |
| `exponent_mu` | 1.5 | 申报书 3.2.1 | `physical_prior.py` |
| `dilation_rates` | [2, 3, 4] | 申报书 3.2.1 | `multiscale.py` |

### 工程实现参数（可调整）

| 参数 | 当前值 | 说明 | 配置位置 |
|------|--------|------|----------|
| `base_channels` | 32 | Encoder 起始通道数 | `encoder.py` |
| `window_size` | 15 | 物理先验局部窗口 | `physical_prior.py` |
| `guided_radius` | 15 | 引导滤波半径 | `guided_filter.py` |
| `guided_eps` | 0.01 | 引导滤波正则化 | `guided_filter.py` |
| `image_size` | 256 | 输入图像尺寸 | `haze_density.yaml` |
| `batch_size` | 4 | 批量大小 | `haze_density.yaml` |

---

## 四、下一步

### 【本地执行】Git 提交

```bash
git add .
git commit -m "feat: 实现雾密度网络核心特征提取模块

- 添加 basic_blocks.py: ConvBlock / DownsampleBlock
- 添加 encoder.py: Encoder 下采样
- 添加 residual_blocks.py: RB / SDRB (dilation=2,3,4)
- 添加 eca.py: ECA 通道注意力
- 添加 multiscale.py: 多尺度分支
- 添加 test_core_modules.py: 测试脚本"
git push
```

### 【在 Colab 执行】核心模块测试

```python
# Step 1: 更新代码
!git pull

# Step 2: 运行核心模块测试
!python scripts/test_core_modules.py
```

### 验收标准

Phase 2 完成需满足：
- [ ] Encoder: shape/dtype/device/finite/backward 全部通过
- [ ] ResidualBlock: 输入输出 shape 一致，梯度正常
- [ ] DilatedResidualBlock: dilation=2,3,4 均正常工作
- [ ] ECA: 输出范围 [0,1]，梯度正常
- [ ] MultiScale: 3 路并行，输出 3*C 通道
- [ ] GPU Test: CUDA 执行正常

---

## 五、文件清单

### 新增文件 (Phase 2)

| 文件 | 说明 |
|------|------|
| `src/models/haze_density/basic_blocks.py` | 基础卷积块 |
| `src/models/haze_density/encoder.py` | Encoder |
| `src/models/haze_density/residual_blocks.py` | RB / SDRB |
| `src/models/haze_density/eca.py` | ECA |
| `src/models/haze_density/multiscale.py` | 多尺度分支 |
| `scripts/test_core_modules.py` | 核心模块测试 |

---

**最后更新**: 2026-08-28
