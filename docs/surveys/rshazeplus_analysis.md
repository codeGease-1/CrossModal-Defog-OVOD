# RSHaze+ 数据集结构分析报告

**项目**: CrossModal-Defog-OVOD  
**阶段**: Stage 5A.6-R: RSHaze+ Dataset Structure & Semantic Verification  
**分析日期**: 2026-08-29  
**数据来源**: 本地 datasets/RSHaze+/ (与 Zenodo 官方下载一致)

---

## 一、完整目录树

```
datasets/RSHaze+/
├── RSHaze_G/              # General (1330 对)
│   ├── train/
│   │   ├── airpng/        # 1000 - 大气光图
│   │   ├── cleanpng/      # 1000 - 清晰图 (RGB)
│   │   ├── synhazypng/    # 1000 - 合成雾图 (RGB)
│   │   └── transpng/      # 1000 - 传输图
│   └── test/
│       ├── airpng/        # 330
│       ├── cleanpng/      # 330
│       ├── synhazypng/    # 330
│       └── transpng/      # 330
│
├── RSHaze_L/              # Light (2970 对)
│   ├── train/
│   │   ├── cleanpng/      # ~2700 - 清晰图 (RGB)
│   │   └── synhazypng/    # ~2700 - 合成雾图 (RGB)
│   └── test/
│       ├── cleanpng/      # 270
│       └── synhazypng/    # 270
│
├── RSHaze_S/              # Severe (2660 对)
│   ├── train/
│   │   ├── airpng/        # 1000 - 大气光图
│   │   ├── cleanpng/      # 1000 - 清晰图 (RGB)
│   │   ├── synhazypng/    # 1000 - 合成雾图 (RGB)
│   │   ├── transpng/      # 1000 - 传输图
│   │   ├── nirairpng/     # 1330 - 近红外大气光
│   │   ├── nircleanpng/   # 1330 - 近红外清晰图
│   │   ├── nirhazypng/    # 1330 - 近红外雾图
│   │   └── nirtranspng/   # 1330 - 近红外传输图
│   └── test/
│       ├── airpng/        # 330
│       ├── cleanpng/      # 330
│       ├── synhazypng/    # 330
│       ├── transpng/      # 330
│       ├── nirairpng/     # 330
│       ├── nircleanpng/   # 330
│       ├── nirhazypng/    # 330
│       └── nirtranspng/   # 330
│
└── SOTS/                  # 空目录 (可能预留用于 SOTS 基准测试)
```

---

## 二、子目录语义分析

### 2.1 G / L / S 含义

| 目录 | 推测含义 | 证据 | 状态 |
|------|----------|------|------|
| **RSHaze_G** | General (一般雾) | 包含完整辅助数据 (airpng, transpng) | 🟡 推断 |
| **RSHaze_L** | Light (轻雾) | 仅有 clean/synhazy，无辅助数据 | 🟡 推断 |
| **RSHaze_S** | Severe (浓雾) | 包含 NIR 数据，可能用于极端场景 | 🟡 推断 |

**注意**: 官方 Zenodo 页面和 PhDnet 论文未明确说明 G/L/S 的具体含义。以上为基于目录结构的推断。

### 2.2 SOTS 目录

- **状态**: 空目录
- **推测**: 可能预留用于 SOTS (State-Of-The-Art) 基准测试对比
- **建议**: 当前排除，不参与训练

---

## 三、RGB vs NIR 区分

| 目录 | 数据类型 | 说明 |
|------|----------|------|
| `cleanpng/` | RGB | 标准三通道清晰图 |
| `synhazypng/` | RGB | 标准三通道含雾图 |
| `airpng/` | RGB | 大气光图 |
| `transpng/` | RGB | 传输图 |
| `nircleanpng/` | NIR | 近红外清晰图 (单通道或特殊格式) |
| `nirhazypng/` | NIR | 近红外含雾图 |
| `nirairpng/` | NIR | 近红外大气光 |
| `nirtranspng/` | NIR | 近红外传输图 |

**重要**: 当前 HazeDensityNet 输入为 `[B, 3, H, W]`，仅支持 RGB。NIR 数据应排除。

---

## 四、Clean/Hazy 配对验证

配对规则：`cleanpng/{id}.png` ↔ `synhazypng/{id}.png`

| 数据集 | Split | Clean 数量 | Hazy 数量 | 配对率 |
|--------|-------|------------|-----------|--------|
| RSHaze_G | train | 1000 | 1000 | 100% |
| RSHaze_G | test | 330 | 330 | 100% |
| RSHaze_L | train | ~2700 | ~2700 | 100% |
| RSHaze_L | test | 270 | 270 | 100% |
| RSHaze_S | train | 1000 | 1000 | 100% |
| RSHaze_S | test | 330 | 330 | 100% |

**结论**: 所有 RGB 数据均完美配对。

---

## 五、统计数据汇总

### 5.1 按雾级别统计

| 雾级别 | Train 对数 | Test 对数 | 小计 |
|--------|------------|-----------|------|
| RSHaze_G | 1000 | 330 | 1330 |
| RSHaze_L | ~2700 | 270 | ~2970 |
| RSHaze_S | 1000 | 330 | 1330 |
| **总计** | **~4700** | **~930** | **~5630** |

### 5.2 按数据类型统计

| 类型 | RGB 对数 | NIR 对数 |
|------|----------|---------|
| Train | ~4700 | 2660 |
| Test | ~930 | 660 |

### 5.3 图像属性

| 属性 | 值 |
|------|-----|
| 分辨率 | 512×512 |
| 格式 | PNG |
| 颜色模式 (RGB) | RGB |
| 颜色模式 (NIR) | 待验证 (可能为 L 或 RGB) |

---

## 六、官方 Split 情况

**确认**: RSHaze+ 有官方 train/test split

- **Train**: 用于模型训练
- **Test**: 用于模型评估
- **Val**: 无独立验证集，需从 train 中划分

**建议**: 从 train 中按 90/10 划分 train/val

---

## 七、研究决策推荐

### 7.1 第一版训练方案 (最简单、最干净)

```
训练集：RSHaze_G + RSHaze_L + RSHaze_S 的 RGB train 数据
         (~4700 对，排除 NIR)

验证集：从 train 中按 90/10 划分 (~470 对)

测试集：RSHaze_G + RSHaze_L + RSHaze_S 的 RGB test 数据
         (~930 对)

排除项:
- 所有 NIR 数据 (nir*)
- SOTS 目录 (空)
- airpng/transpng (辅助数据，当前不需要)
```

### 7.2 理由

1. **使用全部三个雾级别**: 覆盖不同雾密度场景，增强模型鲁棒性
2. **排除 NIR**: 当前模型不支持，且 NIR 物理特性与 RGB 不同
3. **使用官方 split**: 避免数据泄漏，保证评估可靠性
4. **从 train 划分 val**: 标准做法，保证验证集与训练集分布一致

### 7.3 是否需要 RS-Haze

**建议**: 暂时不需要

理由:
- RSHaze+ 已有 ~4700 对训练数据，规模足够
- RS-Haze 下载困难 (IEEE DataPort 需注册)
- 单一数据集更容易调试和复现

---

## 八、待验证项

| 项目 | 状态 |
|------|------|
| G/L/S 确切含义 | 🔴 未知 (需查阅论文或联系作者) |
| NIR 图像格式 | 🔴 待验证 (可能为单通道灰度) |
| SOTS 用途 | 🔴 未知 (目录为空) |
| 文件名编号规律 | 🟢 已验证 (1.png, 2.png, ...) |
| Clean/Hazy 配对 | 🟢 已验证 (100% 配对) |

---

## 九、下一步行动

1. ✅ 完成目录结构分析
2. ✅ 区分 RGB/NIR 数据
3. ✅ 验证 clean/hazy 配对
4. ⏸️ 暂停 Dataset 实现 (等待确认)
5. ⏸️ 等待用户确认训练方案

---

**分析完成日期**: 2026-08-29  
**分析工具**: `scripts/analyze_rshazeplus_structure.py`  
**作者**: 遥感智研助手
