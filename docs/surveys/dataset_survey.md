# 遥感含雾数据集调研与实验数据方案设计

**项目**: CrossModal-Defog-OVOD - 跨模态语义恢复引导的含雾低质遥感影像开放词汇目标检测  
**模块**: 雾密度感知网络 (Haze Density Perception Network)  
**调研日期**: 2026-08-28  
**调研目的**: 为雾密度感知网络确定训练/验证/测试数据方案

---

## 一、研究目标回顾

当前模型**不是传统"去雾网络"**，而是**雾密度感知网络**。

训练目标：
```
Hazy RGB Image
    ↓
Physical Prior (暗通道 + 对比度 + 颜色偏移)
    ↓
S_final (监督信号)
    
HazeDensityNet(Hazy RGB)
    ↓
Predicted Haze Density
    ↓
MSE(Prediction, S_final)
```

**核心需求**：
- 提供真实或合适的含雾遥感 RGB 图像
- **不要求**每张训练图像必须存在 clear ground truth
- 重点：数据集应体现**非均匀雾霾的空间异质性**

---

## 二、数据集对比表

| 属性 | RS-Haze | RSHaze5K | RSHaze+ | RRSHID | RICE1/2 | RESIDE | UAV-Haze |
|------|---------|----------|---------|--------|---------|--------|----------|
| **遥感图像** | ✅ | ✅ | ✅ | ✅ | ✅ | ❌(自然图像) | ✅(UAV) |
| **RGB 格式** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **含雾图像** | ✅ | ✅ | ✅ | ✅ | ✅(云/雾) | ✅ | ✅ |
| **配对清晰图** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌(仅 150 张真实雾图) |
| **雾类型** | 合成非均匀 | 合成非均匀 | 合成非均匀 | **真实非均匀** | 云/薄雾混合 | 合成/真实 | 真实非均匀 |
| **数据来源** | 合成 | 合成 | 合成 | **真实采集** | 合成 | 合成/真实 | 真实采集 |
| **数据规模** | ~500 对 | 5000 对 | ~10000 对 | ~4000 对 | 500+450 组 | 4000+ 对 | 150 张 |
| **分辨率** | 512×512 | 高解析 | 高解析 | 多样 | 512×512 | 多样 | 多样 |
| **Train/Test** | 有 | 有 | 有 | 有 | 有 | 有 | 无 |
| **公开下载** | ✅GitHub | ✅IEEE DataPort | ✅Zenodo | ✅GitHub | ✅GitHub | ✅Google Site | ✅GitHub |
| **下载难度** | 低 | 中(需注册) | **低** | 低 | 低 | 低 | 低 |
| **License** | 学术 | 学术 | CC-BY-4.0 | 学术 | 学术 | 学术 | 学术 |
| **适合雾密度训练** | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | ❌ | ⭐⭐ |
| **适合外部测试** | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | ❌ | ⭐⭐⭐ |

---

## 三、各数据集详细说明

### 3.1 RS-Haze (Remote Sensing Haze Dataset)

**官方来源**:
- GitHub: https://github.com/zhuqinghe/RS-Haze
- IEEE DataPort: https://ieee-dataport.org/documents/rshaze5k

**关键信息**:
- **规模**: ~500 对含雾/清晰图像
- **分辨率**: 512×512
- **类型**: 合成含雾 (基于大气散射模型)
- **特点**: 
  - 早期遥感去雾基准数据集
  - 使用真实云层和雾层作为掩膜合成
  - 非均匀雾分布
- **训练/测试划分**: 有官方划分
- **下载**: GitHub 直接下载，难度低
- **适用性**: 
  - ✅ 适合雾密度感知训练 (非均匀雾)
  - ⚠️ 规模较小，建议与其他数据集联合使用

---

### 3.2 RSHaze5K

**官方来源**:
- IEEE DataPort: https://doi.org/10.21227/w6e5-xq64

**关键信息**:
- **规模**: 5000 对高解析遥感图像
- **类型**: 合成含雾 (使用真实云/雾层 + 专业大气散射模型)
- **特点**:
  - 比 RS-Haze 规模大 10 倍
  - 更高质量的合成方法
  - 非均匀雾分布
- **下载**: 需 IEEE DataPort 注册，难度中等
- **适用性**:
  - ✅✅ 非常适合雾密度感知训练
  - ✅ 规模足够，可独立作为训练集

---

### 3.3 RSHaze+

**官方来源**:
- Zenodo: https://zenodo.org/records/13837162
- DOI: 10.1016/j.inffus.2024.102277
- 关联论文: PhDnet (Information Fusion 2024)

**关键信息**:
- **规模**: ~10000 对 (7.4 GB)
- **类型**: 合成含雾
- **特点**:
  - 目前规模最大的遥感去雾数据集之一
  - 基于 PhDnet 论文
  - 非均匀雾分布
- **下载**: Zenodo 直接下载，难度低
- **License**: CC-BY-4.0
- **适用性**:
  - ✅✅✅ 最适合作为主要训练集
  - ✅ 规模大、下载方便、许可友好

---

### 3.4 RRSHID (Real-World Remote Sensing Hazy Image Dataset)

**官方来源**:
- GitHub: https://github.com/AeroVILab-AHU/RRSHID
- 论文: TGRS 2025, "Real-World Remote Sensing Image Dehazing: Benchmark and Baseline"
- arXiv: https://arxiv.org/abs/2503.17966

**关键信息**:
- **规模**: ~4000 对真实含雾/去雾图像对
- **类型**: **真实世界采集** (非合成)
- **特点**:
  - **首个大规模真实世界遥感含雾配对数据集**
  - 复杂大气条件
  - 严重颜色失真
  - 高度非均匀雾分布
- **下载**: GitHub 直接下载，难度低
- **适用性**:
  - ✅✅✅✅ **最适合作为外部测试集**
  - ✅ 真实数据可检验模型泛化能力
  - ⚠️ 真实数据可能不适合直接训练 (分布差异大)

---

### 3.5 RICE (Remote sensing Image Cloud rEmoving)

**官方来源**:
- GitHub: https://github.com/BUPTLdy/RICE_DATASET
- 论文: arXiv:1901.00600 "A Remote Sensing Image Dataset for Cloud Removal"

**关键信息**:
- **RICE1**: 500 对云/无云图像 (512×512)
- **RICE2**: 450 组 (无云参考图 + 云图 + 云掩膜)
- **类型**: 云移除数据集 (非纯雾)
- **特点**:
  - 主要关注云层而非雾霾
  - 云和雾的物理特性有差异
  - 可作为补充数据
- **下载**: GitHub 直接下载，难度低
- **适用性**:
  - ⚠️ 可作为补充训练数据 (云/雾有部分共性)
  - ❌ 不建议作为主要训练集 (任务定义不同)

---

### 3.6 RESIDE (REalistic Single Image DEhazing)

**官方来源**:
- Website: https://sites.google.com/view/reside-dehaze-datasets
- GitHub: https://github.com/Boyiliee/RESIDE-dataset-link
- 论文: IEEE TIP 2019 "Benchmarking Single-Image Dehazing and Beyond"

**关键信息**:
- **规模**: 4000+ 对 (5 个子集)
- **类型**: 合成/真实含雾自然图像
- **特点**:
  - 最全面的去雾基准数据集
  - **非遥感图像** (地面视角自然场景)
  - 包含 RTTS、ITS、SOTS、HSTS、IHOS 子集
- **下载**: Google Drive / Dropbox / 百度云
- **适用性**:
  - ❌ **不适合** (非遥感图像，视角和场景差异大)
  - ⚠️ 仅可用于跨域泛化能力测试

---

### 3.7 UAV-Haze Dataset

**官方来源**:
- GitHub: https://github.com/Lyndo125/Real-outdoor-UAV-remote-sensing-hazy-dataset

**关键信息**:
- **规模**: 150 张真实户外 UAV 含雾图像
- **类型**: 真实 UAV 采集
- **特点**:
  - **无配对清晰图**
  - 真实非均匀雾
  - 规模太小
- **适用性**:
  - ❌ 不适合训练 (规模太小、无配对)
  - ⚠️ 可用于无参考测试 (定性评估)

---

## 四、重点关注：非均匀雾特性

### 4.1 非均匀雾的重要性

本项目核心创新之一是**"非均匀雾霾的空间感知"**。

理想训练数据应体现：
- ✅ 不同区域雾密度不同
- ✅ 局部浓雾
- ✅ 局部薄雾
- ✅ 空间异质性

### 4.2 数据集非均匀雾评估

| 数据集 | 非均匀雾 | 空间异质性 | 局部浓雾 | 评估 |
|--------|----------|------------|----------|------|
| **RRSHID** | ✅✅✅ | ✅✅✅ | ✅✅✅ | 真实世界，高度非均匀 |
| **RSHaze+** | ✅✅✅ | ✅✅✅ | ✅✅✅ | 合成但基于真实雾层 |
| **RSHaze5K** | ✅✅✅ | ✅✅✅ | ✅✅✅ | 专业大气散射模型 |
| **RS-Haze** | ✅✅ | ✅✅ | ✅✅ | 较早但仍有非均匀性 |
| **RICE** | ✅ | ✅ | ⚠️ | 云为主，雾为辅 |
| **UAV-Haze** | ✅✅✅ | ✅✅✅ | ✅✅✅ | 真实但规模小 |
| **RESIDE** | ✅✅ | ✅✅ | ✅✅ | 自然图像，非遥感 |

---

## 五、推荐实验方案

### 5.1 方案 A (推荐) - 综合方案

```
训练集：RSHaze+ (~10000 对)
         + RS-Haze (500 对，数据增强)
         + RICE1 (500 对，补充云/雾混合场景)
         
验证集：RSHaze+ 官方验证集划分

测试集：RRSHID (真实世界，检验泛化能力)
         + RS-Haze 官方测试集 (合成数据基准对比)
```

**理由**:
1. **RSHaze+** 作为主训练集：
   - 规模最大 (~10000 对)
   - 下载方便 (Zenodo 直接下载)
   - License 友好 (CC-BY-4.0)
   - 非均匀雾特性明显
   - 适合 Colab T4 训练

2. **RS-Haze** 作为补充：
   - 早期基准，便于与现有方法对比
   - 增加数据多样性

3. **RICE1** 作为补充：
   - 云/雾混合场景，增加鲁棒性
   - 物理先验对云也有一定适用性

4. **RRSHID** 作为外部测试集：
   - **真实世界数据**
   - 检验模型泛化能力
   - 体现项目创新价值

---

### 5.2 方案 B (简化) - 单一数据集方案

```
训练集：RSHaze5K (5000 对，使用官方 train 划分)
验证集：RSHaze5K 官方验证划分
测试集：RSHaze5K 官方测试划分 + RRSHID (外部测试)
```

**理由**:
- 简化数据管理
- RSHaze5K 质量高、规模适中
- 仍使用 RRSHID 检验泛化

---

### 5.3 方案 C (最小) - 快速验证方案

```
训练集：RS-Haze (500 对，强数据增强)
验证集：RS-Haze 官方验证划分
测试集：RS-Haze 官方测试集 + RRSHID 子集
```

**理由**:
- 快速验证模型可行性
- 数据量小，训练快
- 适合初期调试

---

## 六、下载难度与 Colab 适配性

### 6.1 下载难度评估

| 数据集 | 下载方式 | 难度 | 时间 | 备注 |
|--------|----------|------|------|------|
| **RSHaze+** | Zenodo 直接下载 | ⭐低 | ~10 分钟 | 7.4GB，需稳定网络 |
| **RS-Haze** | GitHub 直接下载 | ⭐低 | ~5 分钟 | 体积小 |
| **RRSHID** | GitHub / 百度云 | ⭐低 | ~15 分钟 | 提供多种下载方式 |
| **RSHaze5K** | IEEE DataPort | ⭐⭐中 | ~20 分钟 | 需注册账号 |
| **RICE** | GitHub | ⭐低 | ~5 分钟 | 体积小 |

### 6.2 Colab T4 适配性

**Colab T4 限制**:
- GPU 显存：16 GB
- 磁盘空间：~100 GB (临时)
- 会话时长：~12 小时

**推荐方案**:
1. **RSHaze+** (7.4 GB): ✅ 适合，可完整加载
2. **RSHaze5K**: ✅ 适合，规模适中
3. **RS-Haze + RICE**: ✅ 适合，体积小
4. **RRSHID**: ✅ 适合用作测试集

**Patch Crop 建议**:
- 对于大图数据集，建议使用 patch crop 扩充训练样本
- 推荐 patch 大小：256×256 或 512×512
- 可随机 crop + 翻转 + 旋转增强

---

## 七、数据集分布差异分析

### 7.1 合成 vs 真实

| 维度 | 合成数据集 | 真实数据集 |
|------|------------|------------|
| **代表性** | RS-Haze, RSHaze5K, RSHaze+ | RRSHID, UAV-Haze |
| **雾分布** | 可控、可重复 | 复杂、不可控 |
| **配对质量** | 完美配对 | 近似配对 |
| **训练适用性** | ✅ 适合训练 | ⚠️ 适合测试 |
| **域差距** | 合成 - 真实域差距 | 真实世界 |

### 7.2 跨数据集分布差异

- **RS-Haze → RRSHID**: 合成→真实，域差距大
- **RSHaze+ → RRSHID**: 合成→真实，域差距大
- **RICE → RRSHID**: 云→雾，任务差异

**建议**:
- 训练使用合成数据 (RSHaze+)
- 测试使用真实数据 (RRSHID)
- 评估跨域泛化能力

---

## 八、Patch Crop 数据增强建议

### 8.1 是否建议 Patch Crop

**建议**: ✅ 强烈建议

**理由**:
1. **扩充训练样本**: 10000 对 → 100000+ patches
2. **适应 Colab 显存**: 小 patch 可降低显存占用
3. **增加多样性**: 随机 crop 覆盖不同雾密度区域
4. **聚焦局部特征**: 雾密度感知是局部任务

### 8.2 推荐配置

```python
# 推荐 patch 配置
PATCH_SIZE = 256  # 或 512 (如果显存允许)
RANDOM_CROP = True
HORIZONTAL_FLIP = True
VERTICAL_FLIP = True  # 遥感图像可垂直翻转
ROTATION = [0, 90, 180, 270]  # 可选
```

---

## 九、最终推荐方案总结

### 9.1 推荐方案

```
【主训练集】RSHaze+ (~10000 对)
  - 来源：https://zenodo.org/records/13837162
  - 理由：规模最大、下载方便、License 友好、非均匀雾明显
  
【补充训练集】RS-Haze (500 对) + RICE1 (500 对)
  - 理由：增加数据多样性、便于基准对比
  
【验证集】RSHaze+ 官方验证划分
  
【外部测试集】RRSHID (~4000 对真实数据)
  - 来源：https://github.com/AeroVILab-AHU/RRSHID
  - 理由：真实世界数据，检验泛化能力，体现项目价值
```

### 9.2 为什么选择 RSHaze+ 作为主训练集

1. **规模优势**: ~10000 对，足够训练深度网络
2. **下载便利**: Zenodo 直接下载，无需注册
3. **许可友好**: CC-BY-4.0，学术使用无限制
4. **非均匀雾**: 基于真实雾层合成，空间异质性明显
5. **Colab 适配**: 7.4GB，可完整加载到 Colab 磁盘
6. **论文支持**: 关联 PhDnet 论文 (Information Fusion 2024)

### 9.3 为什么选择 RRSHID 作为外部测试集

1. **真实世界**: 首个大规模真实遥感含雾配对数据集
2. **域差距检验**: 合成训练→真实测试，检验泛化
3. **非均匀雾**: 真实复杂大气条件
4. **学术价值**: TGRS 2025 论文，权威性高
5. **项目契合**: 体现"非均匀雾霾空间感知"创新点

---

## 十、下一步行动

### 10.1 数据准备 (【在 Colab 执行】)

```python
# Step 1: 下载 RSHaze+
!wget https://zenodo.org/records/13837162/files/RSHaze+.zip?download=1
!unzip RSHaze+.zip

# Step 2: 下载 RS-Haze
!git clone https://github.com/zhuqinghe/RS-Haze.git

# Step 3: 下载 RRSHID (测试集)
!git clone https://github.com/AeroVILab-AHU/RRSHID.git
```

### 10.2 Dataset 类实现 (下一阶段)

- 创建 `src/data/haze_dataset.py`
- 支持 RSHaze+、RS-Haze、RRSHID
- 支持 patch crop 数据增强
- 支持 train/val/test 划分

### 10.3 训练框架 (后续阶段)

- 创建 `src/losses.py` (MSE Loss)
- 创建 `src/train.py` (训练循环)
- 实现 checkpoint 保存/加载
- 实现 AMP 混合精度训练

---

**文档版本**: v1.0  
**最后更新**: 2026-08-28  
**作者**: 遥感智研助手
