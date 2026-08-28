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
| `requirements.txt` | [OK] | 依赖列表已定义 (10 项) |
| `configs/haze_density.yaml` | [OK] | 配置文件已创建 |
| `docs/project_status.md` | [OK] | 本文档 |
| `README.md` | [OK] | 项目说明已创建 |
| `docs/colab.md` | [OK] | Colab 使用说明已创建 |
| `scripts/setup_colab.py` | [OK] | Colab 环境设置脚本 |
| `scripts/static_check.py` | [OK] | 本地静态检查脚本 |
| `scripts/smoke_test.py` | [OK] | Smoke test 骨架 |
| `scripts/train_haze_density.py` | [OK] | 训练入口骨架 |
| `__init__.py` 导入结构 | [OK] | Python package 结构完整 |

**静态检查结果**: 所有检查通过
- 目录结构：[OK]
- Python 语法：[OK] (9 个文件)
- YAML 语法：[OK] (1 个文件)

### Phase 1: 物理先验模块 [OK] 已完成（待 Colab 验证）

| 任务 | 状态 | 说明 |
|------|------|------|
| `src/models/haze_density/physical_prior.py` | [OK] | Dark Channel / Local Contrast / Color Shift |
| `src/models/haze_density/guided_filter.py` | [OK] | Guided Filtering |
| `scripts/test_physical_prior.py` | [OK] | 测试脚本已创建 |
| `scripts/visualize_physical_prior.py` | [OK] | 可视化脚本已创建 |
| Shape Test | [WAIT] | 【在 Colab 执行】 |
| GPU Test | [WAIT] | 【在 Colab 执行】 |

**本地静态检查**: [OK] 所有 Python 语法检查通过

### Phase 2: Encoder + 基础模块 [WAIT] 待开始

| 任务 | 状态 | 说明 |
|------|------|------|
| `src/models/haze_density/basic_blocks.py` | [WAIT] | ConvBlock / BN / ReLU |
| `src/models/haze_density/encoder.py` | [WAIT] | Encoder 下采样 |

### Phase 3: 残差模块 + ECA [WAIT] 待开始

| 任务 | 状态 | 说明 |
|------|------|------|
| `src/models/haze_density/residual_blocks.py` | [WAIT] | RB / SDRB |
| `src/models/haze_density/eca.py` | [WAIT] | ECA Attention |

### Phase 4: Decoder + 完整模型 [WAIT] 待开始

| 任务 | 状态 | 说明 |
|------|------|------|
| `src/models/haze_density/decoder.py` | [WAIT] | Decoder 上采样 |
| `src/models/haze_density/haze_density_net.py` | [WAIT] | 完整模型组装 |

### Phase 5: 训练框架 [WAIT] 待开始

| 任务 | 状态 | 说明 |
|------|------|------|
| `src/data/haze_dataset.py` | [WAIT] | Dataset 类 |
| `src/losses.py` | [WAIT] | MSE Loss |
| `src/train.py` | [WAIT] | 训练循环 |
| `src/validate.py` | [WAIT] | 验证循环 |

### Phase 6: Colab Smoke Test [WAIT] 待开始

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
| `dilation_rates` | [2, 3, 4] | 申报书 3.2.1 | 待实现 |

### 工程实现参数（可调整）

| 参数 | 当前值 | 说明 | 配置位置 |
|------|--------|------|----------|
| `window_size` | 15 | 物理先验局部窗口 | `physical_prior.py` |
| `guided_radius` | 15 | 引导滤波半径 | `guided_filter.py` |
| `guided_eps` | 0.01 | 引导滤波正则化 | `guided_filter.py` |
| `base_channels` | 32 | Encoder 起始通道数 | `haze_density.yaml` |
| `image_size` | 256 | 输入图像尺寸 | `haze_density.yaml` |
| `batch_size` | 4 | 批量大小 | `haze_density.yaml` |
| `lr` | 1e-4 | 学习率 | `haze_density.yaml` |
| `epochs` | 100 | 训练轮数 | `haze_density.yaml` |

---

## 四、技术债务

| 问题 | 优先级 | 说明 |
|------|--------|------|
| 无 | - | 当前无技术债务 |

---

## 五、下一步

### 【本地执行】Git 提交

```bash
# 1. 查看变更
git status

# 2. 添加文件
git add .

# 3. 提交
git commit -m "feat: 实现物理先验雾密度估计模块

- 添加 physical_prior.py: Dark Channel / Local Contrast / Color Shift
- 添加 guided_filter.py: 引导滤波实现
- 添加 test_physical_prior.py: 测试脚本
- 添加 visualize_physical_prior.py: 可视化脚本
- 所有申报书规定参数已正确实现"

# 4. 推送
git push
```

### 【在 Colab 执行】物理先验测试

```python
# Step 1: 克隆最新代码
!git pull  # 或重新 clone

# Step 2: 安装依赖
!pip install -r requirements.txt

# Step 3: 运行测试
!python scripts/test_physical_prior.py

# Step 4: 运行可视化（可选）
!python scripts/visualize_physical_prior.py --generate-test
```

### 验收标准

Phase 1 完成需满足：
- [ ] 所有 Shape Test 通过
- [ ] 所有 Range Test 通过（输出在 [0, 1]）
- [ ] 所有 Finite Test 通过（无 NaN/Inf）
- [ ] Batch Test 通过（B=1,2,4,8）
- [ ] GPU Test 通过（CUDA 执行）
- [ ] Constructive Test 通过（常数图/渐变图/局部雾图）
- [ ] 可视化结果合理

---

## 六、文件清单

### 新增文件 (Phase 0)

| 文件 | 说明 |
|------|------|
| `requirements.txt` | Python 依赖列表 |
| `configs/haze_density.yaml` | 模型配置 |
| `README.md` | 项目说明 |
| `docs/project_status.md` | 项目状态跟踪 |
| `docs/colab.md` | Colab 使用说明 |
| `scripts/setup_colab.py` | Colab 环境初始化 |
| `scripts/static_check.py` | 本地静态检查 |
| `scripts/smoke_test.py` | Smoke test 骨架 |
| `scripts/train_haze_density.py` | 训练入口骨架 |

### 新增文件 (Phase 1)

| 文件 | 说明 |
|------|------|
| `src/models/haze_density/physical_prior.py` | 物理先验计算 |
| `src/models/haze_density/guided_filter.py` | 引导滤波 |
| `scripts/test_physical_prior.py` | 物理先验测试 |
| `scripts/visualize_physical_prior.py` | 物理先验可视化 |

### 修改文件 (Phase 1)

| 文件 | 说明 |
|------|------|
| `src/models/haze_density/__init__.py` | 添加物理先验模块导出 |
| `docs/project_status.md` | 更新项目状态 |

---

## 七、物理先验模块接口

### 主接口

```python
from src.models.haze_density import generate_s_final

# 输入：image [B, 3, H, W], 范围 [0, 1]
# 输出：s_final [B, 1, H, W], 范围 [0, 1]
s_final = generate_s_final(image)

# 返回中间结果（用于 debug/可视化）
result = generate_s_final(image, return_intermediate=True)
# result 包含：D_hat, C_hat, K_hat, S_hat, S_final
```

### 独立模块接口

```python
from src.models.haze_density import (
    dark_channel,
    local_contrast,
    color_shift,
    compute_physical_prior,
)

# 单独计算各先验
d = dark_channel(image, window_size=15)  # [B, 1, H, W]
c = local_contrast(image, window_size=15)  # [B, 1, H, W]
k = color_shift(image, window_size=15)  # [B, 1, H, W]

# 计算融合结果
d_hat, c_hat, k_hat, s_hat = compute_physical_prior(image, window_size=15)
```

### nn.Module 版本

```python
from src.models.haze_density import PhysicalPriorModule

module = PhysicalPriorModule(window_size=15, guided_radius=15, guided_eps=0.01)
s_final = module(image)
```

---

**最后更新**: 2026-08-28
