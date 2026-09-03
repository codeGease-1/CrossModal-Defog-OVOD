# Stage 6-1: HazeDensityNet Integration Plan

**项目**: CrossModal-Defog-OVOD  
**阶段**: Stage 6-1: Density Map Integration Planning  
**创建日期**: 2026-09-01  
**状态**: ✅ 规划完成，待实施决策

---

## 一、HazeDensityNet 完成状态总结

### 1.1 模型性能

| 指标 | Test Set (RSHaze+) |
|------|-------------------|
| **MAE** | 0.079607 |
| **RMSE** | 0.099042 |
| **Pearson** | 0.872662 |

### 1.2 Checkpoint 信息

| 项目 | 值 |
|------|-----|
| **路径** | `experiments/haze_density/checkpoints/formal/best.pth` |
| **base_channels** | 32 |
| **use_sigmoid** | True |
| **输出范围** | [0, 1] |

### 1.3 模型接口

```python
from src.models.haze_density import HazeDensityNet

# 创建模型
density_net = HazeDensityNet(base_channels=32)

# 前向传播
hazy_image = torch.rand(B, 3, H, W)  # [0, 1]
density_map = density_net(hazy_image)  # [B, 1, H, W], [0, 1]
```

---

## 二、CrossModal-Defog-OVOD 主模型结构分析

### 2.1 当前项目状态

根据 README 和调研报告，本项目目标为：

> **跨模态语义恢复引导的含雾低质遥感影像开放词汇目标检测**

**核心技术路线**（申报书）:
```
雾密度感知网络 → WC-CFAU(Mamba 低频去雾+CNN 高频增强) → 全局特征增强 (VSS+FFT) → ResBlock-PixelShuffle 超分 → SAM+RemoteCLIP 开放词汇分割
```

### 2.2 已实现模块

| 模块 | 状态 | 说明 |
|------|------|------|
| **HazeDensityNet** | ✅ 完成 | 雾密度感知网络 |
| **Physical Prior** | ✅ 完成 | 物理先验监督信号 |
| **RSHaze+ Dataset** | ✅ 完成 | 数据加载系统 |

### 2.3 待实现模块

| 模块 | 状态 | 说明 |
|------|------|------|
| **WC-CFAU** | ⏸️ 待实现 | Mamba 低频去雾+CNN 高频增强 |
| **VSS+FFT** | ⏸️ 待实现 | 全局特征增强 |
| **Super-Resolution** | ⏸️ 待实现 | ResBlock-PixelShuffle 超分 |
| **SAM+RemoteCLIP** | ⏸️ 待实现 | 开放词汇分割 |

### 2.4 调研报告关键发现

根据《调研报告_直接含雾 OVS 路径可行性评估.md》：

| 发现 | 对本项目的启示 |
|------|----------------|
| 传统去雾针对人眼优化，不一定提升 SAM 性能 | 需设计面向分割的去雾模块 |
| 语言提示可直接替代图像增强 | 可考虑 CLIP 提示增强作为备选 |
| 双路融合（含雾 + 去雾）鲁棒性最强 | 推荐保留原始特征路径 |
| SAM 鲁棒性微调比串行去雾更有效 | 需微调 SAM 适应含雾域 |

---

## 三、Density Map 集成方案设计

### 方案 A: RGB + Density Channel Concatenation

#### 3.1.1 架构设计

```
输入：Hazy RGB [B, 3, H, W]
    ↓
HazeDensityNet → Density [B, 1, H, W]
    ↓
Concat: [Hazy RGB, Density] → [B, 4, H, W]
    ↓
Downstream Module (WC-CFAU / SAM Encoder)
```

#### 3.1.2 代码修改点

```python
# 伪代码
class CrossModalDefogOVOD(nn.Module):
    def __init__(self):
        super().__init__()
        self.density_net = HazeDensityNet(base_channels=32)
        self.backbone = ...  # 修改输入通道从 3 到 4
        
    def forward(self, hazy_image, text_prompts):
        # 生成密度图
        with torch.no_grad():  # 密度图固定，不反向传播
            density_map = self.density_net(hazy_image)
        
        # 拼接输入
        concat_input = torch.cat([hazy_image, density_map], dim=1)  # [B, 4, H, W]
        
        # 下游处理
        features = self.backbone(concat_input)
        ...
```

#### 3.1.3 方案分析

| 维度 | 评估 |
|------|------|
| **修改代码量** | 低（仅需修改 backbone 输入层） |
| **参数增加** | +0.5M (HazeDensityNet 冻结) |
| **训练成本** | 低（密度图 pre-compute，不反向传播） |
| **创新性** | 低（常见做法） |
| **风险** | 低（实现简单，易于调试） |
| **推理延迟** | +15-20ms (密度图生成) |

**优点**:
- ✅ 实现最简单
- ✅ 下游模块感知雾密度空间分布
- ✅ 密度图可冻结，训练稳定

**缺点**:
- ❌ 简单拼接，未充分利用密度图语义
- ❌ 增加输入通道可能破坏预训练权重初始化
- ❌ 密度噪声直接传入下游

---

### 方案 B: Density Encoder Branch + Feature Fusion

#### 3.2.1 架构设计

```
                    → Hazy RGB → Backbone → Visual Features →
                                                                     ↓ Fusion → Detection
输入：Hazy RGB                                                 
                    → Density Net → Density Encoder → Density Features →
```

#### 3.2.2 代码修改点

```python
class DensityEncoder(nn.Module):
    """密度图编码器，将密度图编码为多尺度特征"""
    def __init__(self, in_channels=1, out_channels=[64, 128, 256, 512]):
        super().__init__()
        self.stage1 = ConvBlock(1, out_channels[0], stride=4)
        self.stage2 = ConvBlock(out_channels[0], out_channels[1], stride=2)
        self.stage3 = ConvBlock(out_channels[1], out_channels[2], stride=2)
        self.stage4 = ConvBlock(out_channels[2], out_channels[3], stride=2)
    
    def forward(self, density_map):
        f1 = self.stage1(density_map)
        f2 = self.stage2(f1)
        f3 = self.stage3(f2)
        f4 = self.stage4(f3)
        return [f1, f2, f3, f4]  # 多尺度特征

class CrossModalDefogOVOD(nn.Module):
    def __init__(self):
        super().__init__()
        self.density_net = HazeDensityNet(base_channels=32)
        self.density_encoder = DensityEncoder()
        self.visual_backbone = ...  # 保持 3 通道输入
        
    def forward(self, hazy_image, text_prompts):
        # 视觉分支
        visual_features = self.visual_backbone(hazy_image)
        
        # 密度分支
        with torch.no_grad():
            density_map = self.density_net(hazy_image)
        density_features = self.density_encoder(density_map)
        
        # 特征融合
        fused_features = self.fusion_module(visual_features, density_features)
        ...
```

#### 3.2.3 方案分析

| 维度 | 评估 |
|------|------|
| **修改代码量** | 中（需实现 DensityEncoder + FusionModule） |
| **参数增加** | +0.8M (HazeDensityNet + DensityEncoder) |
| **训练成本** | 中（密度分支可冻结，fusion 需训练） |
| **创新性** | 中（双分支融合常见，但密度编码新颖） |
| **风险** | 中（融合策略需调参） |
| **推理延迟** | +25-35ms (密度编码 + 融合) |

**优点**:
- ✅ 视觉和密度特征独立编码，信息保留完整
- ✅ 多尺度融合，适应不同粒度任务
- ✅ 下游 backbone 保持预训练权重

**缺点**:
- ❌ 需设计融合策略（concat/add/attention）
- ❌ 增加融合模块训练复杂度
- ❌ 密度编码器的设计需实验验证

---

### 方案 C: Attention/Gating Based Density Guidance

#### 3.3.1 架构设计

```
输入：Hazy RGB [B, 3, H, W]
    ↓
Backbone → Visual Features [B, C, H, W/32, W/32]
    ↓
HazeDensityNet → Density [B, 1, H, W]
    ↓ Pool/Downsample
Density Guidance [B, 1, H/32, W/32]
    ↓
Attention Map = Sigmoid(Density Guidance × Visual Features)
    ↓
Guided Features = Visual Features × Attention Map
    ↓
Detection Head
```

#### 3.3.2 代码修改点

```python
class DensityGuidanceModule(nn.Module):
    """密度引导注意力模块"""
    def __init__(self, feature_channels):
        super().__init__()
        self.density_proj = nn.Conv2d(1, feature_channels, kernel_size=1)
        self.visual_proj = nn.Conv2d(feature_channels, feature_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))  # 可学习缩放
        
    def forward(self, visual_features, density_map):
        # 下采样密度图到 feature 分辨率
        density_down = F.adaptive_avg_pool2d(density_map, visual_features.shape[2:])
        
        # 投影到 feature 空间
        density_proj = self.density_proj(density_down)  # [B, C, H', W']
        visual_proj = self.visual_proj(visual_features)
        
        # 生成注意力图
        attention = torch.sigmoid(density_proj + visual_proj)
        
        # 残差连接
        guided_features = visual_features + self.gamma * visual_features * attention
        return guided_features

class CrossModalDefogOVOD(nn.Module):
    def __init__(self):
        super().__init__()
        self.density_net = HazeDensityNet(base_channels=32)
        self.backbone = ...
        self.guidance_modules = nn.ModuleList([
            DensityGuidanceModule(ch) for ch in [256, 512, 1024, 2048]
        ])
        
    def forward(self, hazy_image, text_prompts):
        # 生成密度图
        with torch.no_grad():
            density_map = self.density_net(hazy_image)
        
        # Backbone 多尺度特征
        visual_features = self.backbone(hazy_image)  # List of [B, Ci, Hi, Wi]
        
        # 密度引导
        guided_features = []
        for i, (vf, guidance) in enumerate(zip(visual_features, self.guidance_modules)):
            gf = guidance(vf, density_map)
            guided_features.append(gf)
        
        # 检测头
        ...
```

#### 3.3.3 方案分析

| 维度 | 评估 |
|------|------|
| **修改代码量** | 中（需实现 GuidanceModule，插入 backbone 中间） |
| **参数增加** | +0.6M (HazeDensityNet + GuidanceModules) |
| **训练成本** | 中（guidance 模块需训练，密度图冻结） |
| **创新性** | 高（密度引导注意力，符合申报书"雾密度感知"理念） |
| **风险** | 中（注意力权重需合理初始化） |
| **推理延迟** | +20-30ms (密度投影 + 注意力计算) |

**优点**:
- ✅ 密度图作为注意力引导，语义明确
- ✅ 可学习缩放因子，自适应调整引导强度
- ✅ 残差连接，保证信息不丢失
- ✅ 符合申报书"雾密度感知网络"设计

**缺点**:
- ❌ 需插入 backbone 中间，修改现有结构
- ❌ 注意力机制可能引入训练不稳定
- ❌ 需调参 gamma 初始化

---

## 四、方案对比总结

### 4.1 综合对比

| 维度 | 方案 A (Concat) | 方案 B (Encoder+Fusion) | 方案 C (Attention) |
|------|----------------|------------------------|-------------------|
| **修改代码量** | 低 ⭐⭐⭐ | 中 ⭐⭐ | 中 ⭐⭐ |
| **参数增加** | +0.5M | +0.8M | +0.6M |
| **训练成本** | 低 ⭐⭐⭐ | 中 ⭐⭐ | 中 ⭐⭐ |
| **创新性** | 低 ⭐ | 中 ⭐⭐ | 高 ⭐⭐⭐ |
| **风险** | 低 ⭐⭐⭐ | 中 ⭐⭐ | 中 ⭐⭐ |
| **推理延迟** | +15-20ms | +25-35ms | +20-30ms |
| **符合申报书** | 中 ⭐⭐ | 中 ⭐⭐ | 高 ⭐⭐⭐ |

### 4.2 适用场景

| 方案 | 适用场景 |
|------|----------|
| **方案 A** | 快速验证、基线实现、计算资源受限 |
| **方案 B** | 追求精度、有充足训练资源、需多尺度融合 |
| **方案 C** | 符合申报书理念、需解释性、密度引导是关键创新 |

---

## 五、推荐方案

### 5.1 推荐：方案 C (Attention/Gating Based Density Guidance)

**推荐理由**:

1. **符合申报书设计理念**
   - 申报书强调"雾密度感知网络"作为引导模块
   - Attention 机制实现"感知→引导"的语义映射

2. **创新性与可行性平衡**
   - 创新性高于简单 concat
   - 实现复杂度可控，风险可管理

3. **训练稳定性**
   - 密度图冻结，不反向传播
   - 残差连接保证信息不丢失
   - 可学习 gamma 自适应调整引导强度

4. **可扩展性**
   - 可轻松扩展为多尺度引导
   - 可与后续 Mamba/VSS 模块结合

### 5.2 备选：方案 A (Concat) 作为基线

**建议**: 先实现方案 A 作为快速基线，验证密度图有效性，再升级到方案 C。

---

## 六、下一阶段实施计划

### 6.1 Stage 6-2: Baseline Implementation (方案 A)

| 任务 | 预计时间 |
|------|----------|
| 修改 backbone 输入层 (3→4 通道) | 1 天 |
| 集成 HazeDensityNet (冻结) | 1 天 |
| 验证 forward 流程 | 1 天 |
| 基线训练 + 评估 | 2 天 |
| **合计** | **5 天** |

### 6.2 Stage 6-3: Attention Guidance Implementation (方案 C)

| 任务 | 预计时间 |
|------|----------|
| 实现 DensityGuidanceModule | 2 天 |
| 插入 backbone 多尺度位置 | 2 天 |
| 验证 forward + 梯度流 | 1 天 |
| 训练 + 调参 (gamma 初始化) | 3 天 |
| 与 baseline 对比评估 | 1 天 |
| **合计** | **9 天** |

### 6.3 Stage 6-4: WC-CFAU Integration

| 任务 | 预计时间 |
|------|----------|
| 实现 Mamba 低频去雾分支 | 5 天 |
| 实现 CNN 高频增强分支 | 3 天 |
| 小波变换融合 (DWT/IDWT) | 2 天 |
| 与密度引导模块联调 | 2 天 |
| **合计** | **12 天** |

---

## 七、技术债务与风险

### 7.1 技术债务

| 债务 | 优先级 | 说明 |
|------|--------|------|
| HazeDensityNet 未与下游联合训练 | 中 | 当前冻结，可能未达最优 |
| 密度图分辨率与 feature 不匹配 | 低 | 需 downsample，可能丢失细节 |
| 未考虑密度图不确定性 | 低 | 当前使用 point estimate |

### 7.2 风险与应对

| 风险 | 可能性 | 影响 | 应对策略 |
|------|--------|------|----------|
| 密度噪声传播到下游 | 中 | 中 | 方案 C 的 gamma 可抑制噪声 |
| Backbone 预训练权重破坏 | 低 | 高 | 方案 C 保持 3 通道输入 |
| 训练不稳定 | 中 | 中 | 残差连接 + 渐进式训练 |
| 推理延迟过高 | 低 | 中 | 密度图可 batch pre-compute |

---

## 八、验收标准

### 8.1 Stage 6-2 (Baseline)

| 指标 | 目标 |
|------|------|
| Forward 正常 | ✅ |
| 无 NaN/Inf | ✅ |
| 密度图正确拼接 | ✅ |
| 基线性能 > 无密度图 | ✅ |

### 8.2 Stage 6-3 (Attention)

| 指标 | 目标 |
|------|------|
| Forward 正常 | ✅ |
| 梯度流正确 | ✅ |
| 性能 > Baseline | ✅ |
| 注意力图可视化合理 | ✅ |

---

## 九、结论

**推荐方案**: 方案 C (Attention/Gating Based Density Guidance)

**实施策略**:
1. 先实现方案 A 作为快速基线（5 天）
2. 验证密度图有效性后升级到方案 C（9 天）
3. 后续集成 WC-CFAU 模块（12 天）

**总预计时间**: 26 天（不含后续 SAM+RemoteCLIP 集成）

---

**报告生成日期**: 2026-09-01  
**作者**: 遥感智研助手
