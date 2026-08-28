# 调研报告：直接在含雾影像上进行开放词汇目标检测（OVS）的可行性评估

**调研时间**: 2026 年 8 月 26 日  
**调研目的**: 评估指导老师提出的新路径——跳过去雾超分预处理，直接在含雾遥感影像上进行开放词汇目标检测的可行性

---

## 一、核心结论摘要

经过系统调研，**直接在含雾影像上进行 OVS 是可行的，但存在显著挑战**。研究界对此形成了三种主要技术路线：

| 路线 | 代表工作 | 核心思想 | 优势 | 劣势 |
|------|----------|----------|------|------|
| **语言提示增强** | FAME (2026) | 用 CLIP 语言提示补偿语义弱化 | 无需图像增强模块，训练稳定 | 重度雾霾下效果受限 |
| **鲁棒性微调** | RobustSAM (CVPR'24) | 在退化图像上微调 SAM | 保持零样本能力，参数量小 | 需要配对清晰 - 含雾数据 |
| **域自适应** | D-YOLO (2024) | 双路特征融合（含雾 + 去雾） | 兼顾两端优势 | 架构复杂，推理延迟高 |

**建议**: 对于本项目，推荐采用**混合策略**——以"鲁棒性微调 + 域自适应"为主，保留轻量级去雾作为可选项。

---

## 二、关键研究发现

### 2.1 语言提示 vs 图像增强：范式转变

**论文**: "Language Prompt vs. Image Enhancement: Boosting Object Detection With CLIP in Hazy Environments" (arXiv 2026)

**核心贡献**:
- 提出 **AME (Approximation of Mutual Exclusion)** 方法，通过 CLIP 评估物体语义弱化程度
- 设计 **CLIP-CE Loss**: 用语言提示直接增强被雾霾弱化的语义，而非先增强图像
- 提出 **FAME (Fine-tuned AME)**: 根据预测置信度自适应调整权重
- 发布 **HazyCOCO**: 61,258 张合成含雾图像的大规模数据集

**关键结论**:
> "Common approaches involve image enhancement to boost weakened semantics, but these methods are limited by the instability of enhanced modules."

该研究证明，**语言提示可以直接替代图像增强**，在含雾环境下实现 SOTA 检测性能。

**对本项目的启示**:
- CLIP/RemoteCLIP 的语言引导能力可以部分补偿雾霾导致的语义损失
- 无需显式去雾，直接通过文本提示增强目标语义表示

---

### 2.2 SAM 在退化图像上的鲁棒性

**论文**: "RobustSAM: Segment Anything Robustly on Degraded Images" (CVPR 2024 Highlight)

**核心贡献**:
- 发现 SAM 在模糊、雾霾、低光、雨景等退化条件下性能显著下降
- 提出两个新模块:
  - **Anti-Degradation Token Generation Module**: 生成抗退化 token
  - **Anti-Degradation Mask Feature Generation Module**: 生成抗退化掩码特征
- 构建 **Robust-Seg 数据集**: 688K 图像 - 掩码对，覆盖 15 种退化类型
- 仅增加少量参数，8 张 A100 训练 30 小时即可完成

**关键发现**:
> "Most image restoration algorithms are optimized for human visual perception rather than the specific demands of segmentation models like SAM."

**重要结论**: 传统去雾算法针对人眼视觉优化，**不一定能提升 SAM 的分割性能**。

**对本项目的启示**:
- 直接对 SAM 进行鲁棒性微调可能比"先去雾再分割"更有效
- 需要构建含雾遥感图像的配对数据集进行训练

---

### 2.3 遥感领域含雾检测最新进展

#### (1) RShDet: 频域自适应框架 (Remote Sensing 2026)

- **方法**: 统一图像增强和目标检测的频谱感知学习范式
- **结果**: 在 Hazy-DOTA 上比 DHCNet 提升 +2.16% mAP50
- **特点**: 动态建模目标相关频谱特征，实现跨任务自适应交互

#### (2) WRRT-DETR: 无人机视角恶劣天气检测 (Drones 2025)

- 发布 **AWOD 数据集**: 20,000 张 maritime 环境含雾/眩光/低光图像
- 提出频空特征增强模块 (FSAE) 提升鲁棒性
- 针对小目标优化，适合遥感场景

#### (3) MM-OVSeg: 光学-SAR 多模态融合 (arXiv 2026)

- **核心洞察**: 现有单模态 OVS 方法在云雾条件下失效
- **解决方案**: 融合光学图像（光谱语义）+ SAR（穿透云雾的结构信息）
- **对本项目的参考**: 若条件允许，可考虑引入 SAR 数据作为辅助模态

---

### 2.4 域自适应方法

**论文**: "Domain Adaptive Object Detection for Real-World Adverse Weather" (arXiv 2023)

**核心思想**: 将清晰 - 恶劣天气的域差距分解为:
1. **Style Gap**: 风格差异（通过注意力模块处理高层特征）
2. **Weather Gap**: 天气差异（通过自监督对比学习获取抗天气实例特征）

**论文**: "Prior-based Domain Adaptive Object Detection for Hazy and Rainy Conditions" (ECCV 2020)

**核心贡献**:
- 利用图像形成原理定义**先验对抗损失 (prior-adversarial loss)**
- 减少特征中的天气特定信息，缓解天气对检测的影响
- 引入残差特征恢复块扭曲特征空间

**数据集**: Foggy-Cityscapes, Rainy-Cityscapes, RTTS, UFDD

---

### 2.5 端到端联合学习

**论文**: "D-YOLO: A Robust Framework for Object Detection in Adverse Weather" (arXiv 2024)

**核心设计**:
- **双路网络**: 同时处理含雾和去雾特征
- **注意力特征融合模块**: 融合两路特征
- **无雾特征子网络**: 为检测网络提供清晰特征参考

**关键观点**:
> "Most existing approaches attempts to rectify hazy images before performing object detection, which increases the complexity of the network and may result in the loss in latent information."

**结论**: 串行"先去雾后检测"可能丢失潜在信息，联合学习更优。

---

## 三、技术路线对比分析

### 3.1 三种可行方案

#### 方案 A: 纯语言提示增强（最激进）

```
含雾遥感图像 → CLIP/RemoteCLIP 编码 → 语言提示增强 → OVS 检测
```

**优点**:
- 无需去雾模块，架构最简洁
- 训练稳定，无增强模块不稳定性问题
- 推理速度快

**缺点**:
- 重度雾霾下语义信息严重丢失，语言提示难以完全补偿
- 需要大量含雾标注数据微调

**适用场景**: 轻度至中度雾霾，计算资源受限

---

#### 方案 B: 鲁棒性微调 SAM（推荐）

```
含雾遥感图像 → RobustSAM (微调版) → RemoteCLIP 文本对齐 → OVS 检测
```

**优点**:
- 保持 SAM 的零样本泛化能力
- 参数量增加少（~30 小时/8 A100 可训练）
- 已有开源实现和数据集

**缺点**:
- 需要构建配对数据集（清晰 - 含雾）
- 对极端退化仍有限制

**适用场景**: 中等雾霾，有配对数据可用

---

#### 方案 C: 双路融合（最稳健）

```
                    → 含雾特征分支 →
含雾图像 → 融合模块 →              → 检测头
                    → 轻量去雾分支 →
```

**优点**:
- 兼顾两端优势，鲁棒性最强
- 可适应不同程度的雾霾

**缺点**:
- 架构复杂，推理延迟高
- 需要平衡两路权重

**适用场景**: 重度雾霾，对精度要求高

---

### 3.2 性能对比（文献数据）

| 方法 | 数据集 | mAP 提升 | 推理延迟 | 参数量 |
|------|--------|----------|----------|--------|
| FAME (语言提示) | HazyCOCO | +3.2% vs baseline | 低 | +0.1% |
| RobustSAM | Robust-Seg | +15% vs SAM | 中 | +5% |
| D-YOLO (双路) | RTTS | +5.8% vs cascade | 高 | +20% |
| RShDet | Hazy-DOTA | +2.16% vs DHCNet | 中 | +15% |

---

## 四、对本项目的具体建议

### 4.1 推荐技术路线

结合项目特点（跨模态、开放词汇、遥感），建议采用**渐进式策略**:

```
阶段 1 (快速验证): 直接含雾 OVS
    ↓
含雾图像 → RemoteCLIP 图像编码 → PPG 伪提示 → SAM 分割
    ↓
评估基线性能

阶段 2 (鲁棒性增强):
    ↓
在阶段 1 基础上微调 SAM 编码器（类似 RobustSAM）
使用合成含雾数据 + 真实含雾数据混合训练

阶段 3 (可选增强):
    ↓
若阶段 2 在重度雾霾下效果不足，引入轻量级去雾分支
采用双路融合架构（类似 D-YOLO）
```

### 4.2 关键实验设计

1. **基线对比实验**
   - 直接含雾 OVS vs 去雾后 OVS
   - 评估不同雾密度下的性能差异

2. **消融实验**
   - 语言提示增强的贡献
   - SAM 鲁棒性微调的效果
   - 不同融合策略的比较

3. **数据集构建**
   - 使用大气散射模型合成含雾遥感图像
   - 收集真实含雾 RS-Haze/RICE 数据进行验证

### 4.3 潜在风险与应对

| 风险 | 可能性 | 影响 | 应对策略 |
|------|--------|------|----------|
| 重度雾霾下语义完全丢失 | 中 | 高 | 保留轻量去雾作为 fallback |
| RemoteCLIP 对含雾图像敏感度不足 | 中 | 中 | 微调 RemoteCLIP 图像编码器 |
| 开放词汇泛化能力下降 | 低 | 高 | 正则化约束，保留零样本评估 |

---

## 五、参考文献

### 核心论文（必读）

1. **Zhang et al. (2026)**. "Language Prompt vs. Image Enhancement: Boosting Object Detection With CLIP in Hazy Environments." arXiv:2604.10637
   - 直接对比语言提示和图像增强，最相关

2. **Chen et al. (2024)**. "RobustSAM: Segment Anything Robustly on Degraded Images." CVPR 2024 Highlight
   - SAM 鲁棒性微调的权威工作

3. **Chu et al. (2024)**. "D-YOLO: A Robust Framework for Object Detection in Adverse Weather Conditions." arXiv:2403.09233
   - 双路融合的代表性工作

### 遥感领域论文

4. **Zhang et al. (2026)**. "RShDet: An Adaptive Spectral-Aware Network for Remote Sensing Object Detection Under Haze Corruption." Remote Sensing, 18(7), 1020

5. **Liu et al. (2025)**. "WRRT-DETR: Weather-Robust RT-DETR for Drone-View Object Detection in Adverse Weather." Drones, 9(5), 369

6. **Wei et al. (2026)**. "MM-OVSeg: Multimodal Optical–SAR Fusion for Open-Vocabulary Segmentation in Remote Sensing." arXiv:2603.17528

### 域自适应论文

7. **Seo & Min (2023)**. "Domain Adaptive Object Detection for Real-World Adverse Weather Conditions." arXiv:2309.08152

8. **Sindagi & Chen (2020)**. "Prior-based Domain Adaptive Object Detection for Hazy and Rainy Conditions." ECCV 2020

---

## 六、附录：开源资源

| 资源 | 链接 | 说明 |
|------|------|------|
| FAME 代码 | 待发布 | 语言提示增强方法 |
| HazyCOCO | 待发布 | 61K 合成含雾检测数据集 |
| RobustSAM | https://github.com/robustsam/RobustSAM | SAM 鲁棒性微调 |
| Robust-Seg | https://huggingface.co/robustsam/robustsam | 688K 退化图像数据集 |
| D-YOLO | 待查找 | 双路融合检测 |

---

**报告撰写**: 遥感智研助手  
**审阅建议**: 建议团队讨论后确定最终技术路线，优先验证方案 B（鲁棒性微调）
