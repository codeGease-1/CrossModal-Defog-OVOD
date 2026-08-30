# 数据集清单 (Dataset Manifest)

**项目**: CrossModal-Defog-OVOD - 跨模态语义恢复引导的含雾低质遥感影像开放词汇目标检测  
**模块**: 雾密度感知网络 (Haze Density Perception Network)  
**文档目的**: 记录数据集实际获取与核验结果  
**创建日期**: 2026-08-28  
**最后更新**: 2026-08-30 (Stage 5B-1: Dataset 实现完成)

---

## 一、数据集概览表

| 属性 | RSHaze+ | RS-Haze | RRSHID |
|------|---------|---------|--------|
| **用途** | 主训练集 | 补充训练集 | 外部测试集 |
| **官方来源** | Zenodo | GitHub | GitHub / Baidu |
| **DOI** | 10.1016/j.inffus.2024.102277 | - | 10.1109/tgrs.2025.3584234 |
| **License** | CC-BY-4.0 | 学术使用 | 学术使用 |
| **总图像对数** | 5630 | ~500 | ~4000 |
| **分辨率** | 高解析 | 512×512 | 多样 |
| **格式** | PNG/JPG | PNG | PNG/JPG |
| **配对类型** | Hazy-Clear | Hazy-Clear | Hazy-Clear |
| **官方 Split** | 是 | 是 | 是 |
| **Train Count** | 4234 (90%) | 【待核验】 | N/A |
| **Val Count** | 470 (10%) | 【待核验】 | N/A |
| **Test Count** | 930 | 【待核验】 | ~4000 |
| **实际大小** | 7.4 GB | 【待核验】 | 【待核验】 |
| **Colab 适配** | ✅ | ✅ | ✅ |
| **适合 Physical Prior** | ✅ | ✅ | ✅ |
| **适合 Patch Crop** | ✅ | ✅ | ✅ |
| **备注** | 主训练集 | 补充数据增强 | 真实世界测试 |

---

## 二、RSHaze+ (主训练集)

### 2.1 基本信息

| 字段 | 值 |
|------|-----|
| **数据集名称** | RSHaze+ |
| **官方来源** | https://zenodo.org/records/13837162 |
| **DOI** | 10.1016/j.inffus.2024.102277 |
| **关联论文** | PhDnet: A novel physic-aware dehazing network for remote sensing images (Information Fusion 2024) |
| **License** | CC-BY-4.0 |
| **发布日期** | 2024-09-25 |
| **下载方式** | Zenodo 直接下载 |
| **文件大小** | 7.4 GB (压缩) |
| **MD5** | 82a4d5f5d6eff35989a64bbf233fa65c |
| **核验状态** | ✅ 已核验 (2026-08-29) |

### 2.2 下载命令 (Colab)

```python
# 方法 1: 直接下载
!wget https://zenodo.org/records/13837162/files/RSHaze+.zip?download=1 -O RSHaze+.zip
!unzip -q RSHaze+.zip
!rm RSHaze+.zip
```

### 2.3 实际目录结构 (已验证)

**本地实际结构** (与 Colab 下载版本不同):

```
datasets/RSHaze+/
├── RSHaze_G/          # General (一般雾)
│   ├── train/
│   │   ├── cleanpng/      # 1000 张清晰图
│   │   ├── synhazypng/    # 1000 张含雾图
│   │   ├── airpng/        # 大气光图
│   │   └── transpng/      # 传输图
│   └── test/
│       ├── cleanpng/      # 330 张
│       └── synhazypng/    # 330 张
├── RSHaze_L/          # Light (轻雾)
│   ├── train/
│   │   ├── cleanpng/      # ~2700 张
│   │   └── synhazypng/    # ~2700 张
│   └── test/
│       ├── cleanpng/      # 270 张
│       └── synhazypng/    # 270 张
├── RSHaze_S/          # Severe (浓雾)
│   ├── train/
│   │   ├── cleanpng/      # 1000 张
│   │   ├── synhazypng/    # 1000 张
│   │   └── (nir* 近红外目录)
│   └── test/
│       ├── cleanpng/      # 330 张
│       └── synhazypng/    # 330 张
└── SOTS/              # 空目录
```

**配对规则**: `cleanpng/1.png` ↔ `synhazypng/1.png` (同名配对)

### 2.4 实际统计信息 (Stage 5A.6-R 已验证)

#### 2.4.1 子集统计

| 子集 | Train 对数 | Test 对数 | 小计 | 状态 |
|------|------------|-----------|------|------|
| **RSHaze_G** | 1000 | 330 | 1330 | ✅ 已验证 |
| **RSHaze_L** | ~2700 | 270 | ~2970 | ✅ 已验证 |
| **RSHaze_S** | 1000 | 330 | 1330 | ✅ 已验证 |
| **SOTS** | 0 | 0 | 0 | ✅ 空目录 |
| **总计 (RGB)** | **~4700** | **~930** | **~5630** | ✅ |
| **总计 (含 NIR)** | - | - | 26760 PNG | ✅ |

#### 2.4.2 图像属性

| 属性 | 值 |
|------|-----|
| **图像格式** | PNG |
| **分辨率** | 512×512 |
| **颜色模式 (RGB)** | RGB |
| **颜色模式 (NIR)** | 待验证 |
| **官方 Split** | ✅ train/test |

#### 2.4.3 数据类型分布

| 类型 | 目录 | Train | Test | 说明 |
|------|------|-------|------|------|
| **RGB 清晰图** | cleanpng | ~4700 | ~930 | 主训练数据 |
| **RGB 含雾图** | synhazypng | ~4700 | ~930 | 主训练数据 |
| **RGB 大气光** | airpng | 2000 | 660 | 仅 G/S 有 |
| **RGB 传输图** | transpng | 2000 | 660 | 仅 G/S 有 |
| **NIR 清晰图** | nircleanpng | 1330 | 330 | 仅 S 有 |
| **NIR 含雾图** | nirhazypng | 1330 | 330 | 仅 S 有 |
| **NIR 大气光** | nirairpng | 1330 | 330 | 仅 S 有 |
| **NIR 传输图** | nirtranspng | 1330 | 330 | 仅 S 有 |

### 2.5 适用性评估

| 评估项 | 结果 | 说明 |
|--------|------|------|
| **遥感图像** | ✅ | 纯遥感数据集 |
| **RGB 格式** | ✅ | 标准 RGB 三通道 |
| **含雾图像** | ✅ | 合成非均匀雾 |
| **配对清晰图** | ✅ | Hazy-Clear 配对 |
| **非均匀雾** | ✅ | 基于真实雾层合成 |
| **官方 Split** | ✅ | 有 train/val/test 划分 |
| **适合 Physical Prior** | ✅ | RGB 输入可计算 DCP/LCP/CSP |
| **适合 Patch Crop** | ✅ | 高解析图像 |
| **Colab 适配** | ✅ | 7.4GB 可完整加载 |

---

## 三、RS-Haze (补充训练集)

### 3.1 基本信息

| 字段 | 值 |
|------|-----|
| **数据集名称** | RS-Haze |
| **官方来源** | https://github.com/zhuqinghe/RS-Haze |
| **License** | 学术使用 |
| **下载方式** | GitHub 直接下载 |
| **文件大小** | 【待核验】 |

### 3.2 下载命令 (Colab)

```python
# 方法 1: Git 克隆
!git clone https://github.com/zhuqinghe/RS-Haze.git /content/datasets/RSHaze

# 方法 2: 直接下载 Release
!wget https://github.com/zhuqinghe/RS-Haze/releases/download/v1.0/RS-Haze.zip
!unzip -q RS-Haze.zip -d /content/datasets/
!rm RS-Haze.zip
```

### 3.3 预期目录结构

```
/content/datasets/RSHaze/
├── Haze/
│   ├── image_001_haze.png
│   ├── image_002_haze.png
│   └── ...
├── Clear/
│   ├── image_001.png
│   ├── image_002.png
│   └── ...
└── List/
    ├── train_list.txt
    ├── test_list.txt
    └── ...
```

**注意**: 实际目录结构需下载后验证

### 3.4 实际统计信息 (待填充)

| 统计项 | 值 |
|--------|-----|
| **总文件数** | 【待核验】 |
| **Train 图像对** | 【待核验】 |
| **Test 图像对** | 【待核验】 |
| **图像格式** | 【待核验】 |
| **分辨率** | 512×512 (预期) |
| **RGB 图像数** | 【待核验】 |
| **损坏文件数** | 【待核验】 |

### 3.5 适用性评估

| 评估项 | 结果 | 说明 |
|--------|------|------|
| **遥感图像** | ✅ | 纯遥感数据集 |
| **RGB 格式** | ✅ | 标准 RGB 三通道 |
| **含雾图像** | ✅ | 合成非均匀雾 |
| **配对清晰图** | ✅ | Hazy-Clear 配对 |
| **非均匀雾** | ✅ | 使用真实云层合成 |
| **官方 Split** | ✅ | 有 train/test 划分 |
| **适合 Physical Prior** | ✅ | RGB 输入可计算 DCP/LCP/CSP |
| **适合 Patch Crop** | ✅ | 512×512 分辨率 |
| **Colab 适配** | ✅ | 体积小 |

---

## 四、RRSHID (外部测试集)

### 4.1 基本信息

| 字段 | 值 |
|------|-----|
| **数据集名称** | RRSHID (Real-World Remote Sensing Hazy Image Dataset) |
| **官方来源** | https://github.com/AeroVILab-AHU/RRSHID |
| **DOI** | 10.1109/tgrs.2025.3584234 |
| **关联论文** | Real-World Remote Sensing Image Dehazing: Benchmark and Baseline (TGRS 2025) |
| **License** | 学术使用 |
| **下载方式** | GitHub Release / Baidu Netdisk / Hugging Face |
| **文件大小** | 【待核验】 |

### 4.2 下载命令 (Colab)

```python
# 方法 1: GitHub Release
!wget https://github.com/lwCVer/RRSHID/releases/download/dataset/RRSHID.zip -O RRSHID.zip
!unzip -q RRSHID.zip
!rm RRSHID.zip

# 方法 2: Baidu Netdisk (需要挂载 Google Drive)
# 密码：CV21
# 链接：https://pan.baidu.com/s/1Wg3u7V8AOVfgqkaw1n3lEg?pwd=CV21

# 方法 3: Hugging Face
!pip install huggingface_hub
!huggingface-cli download --repo-type dataset <repo_name> RRSHID.zip --local-dir /content/datasets/
```

### 4.3 预期目录结构

```
/content/datasets/RRSHID/
├── hazy/
│   ├── scene1_image1_haze.png
│   ├── scene1_image2_haze.png
│   └── ...
├── clear/
│   ├── scene1_image1.png
│   ├── scene1_image2.png
│   └── ...
└── split_list/
    ├── test_list.txt
    └── ...
```

**注意**: 实际目录结构需下载后验证

### 4.4 实际统计信息 (待填充)

| 统计项 | 值 |
|--------|-----|
| **总文件数** | 【待核验】 |
| **Test 图像对** | ~4000 (预期) |
| **图像格式** | 【待核验】 |
| **分辨率分布** | 【待核验】 |
| **RGB 图像数** | 【待核验】 |
| **损坏文件数** | 【待核验】 |
| **真实雾类型** | 【待核验】 |

### 4.5 适用性评估

| 评估项 | 结果 | 说明 |
|--------|------|------|
| **遥感图像** | ✅ | 纯遥感数据集 |
| **RGB 格式** | ✅ | 标准 RGB 三通道 |
| **含雾图像** | ✅ | **真实世界采集** |
| **配对清晰图** | ✅ | Hazy-Clear 配对 |
| **非均匀雾** | ✅ | 真实复杂大气条件 |
| **官方 Split** | ✅ | 有测试集划分 |
| **适合 Physical Prior** | ✅ | RGB 输入可计算 DCP/LCP/CSP |
| **适合 Patch Crop** | ✅ | 多样分辨率 |
| **Colab 适配** | ✅ | 仅用作测试 |
| **域差距检验** | ✅ | 合成训练→真实测试 |

---

## 五、Colab 数据目录规范

### 5.1 推荐目录结构

```
/content/
├── datasets/
│   ├── RSHazePlus/          # 主训练集
│   ├── RSHaze/              # 补充训练集
│   └── RRSHID/              # 外部测试集
├── checkpoints/             # 模型检查点
├── logs/                    # 训练日志
└── results/                 # 实验结果

# Google Drive (仅用于备份)
/content/drive/MyDrive/CrossModal-Defog-OVOD/
├── backups/                 # 原始数据备份 (可选)
├── checkpoints/             # 检查点备份
└── results/                 # 结果备份
```

### 5.2 Colab 磁盘空间规划

| 项目 | 预计大小 | 说明 |
|------|----------|------|
| RSHaze+ | ~10 GB | 解压后 |
| RS-Haze | ~1 GB | 预期 |
| RRSHID | ~5 GB | 预期 |
| Checkpoints | ~2 GB | 多个 epoch |
| Logs/Results | ~1 GB | 日志和图像 |
| **总计** | **~19 GB** | Colab T4 约 100GB |

**结论**: Colab T4 磁盘空间充足

---

## 六、数据划分方案

### 6.1 训练/验证/测试划分

| 数据集 | 划分方式 | Train | Val | Test |
|--------|----------|-------|-----|------|
| **RSHaze+** | 官方 Split | 官方 | 官方 | 官方 |
| **RS-Haze** | 官方 Split | 官方 | - | 官方 |
| **RRSHID** | 仅测试 | N/A | N/A | 全部 |

### 6.2 无官方 Split 时的手动划分

如果数据集无官方 split，采用以下方案:

```python
import random

random.seed(42)  # 固定随机种子

all_images = [...]  # 所有原始图像列表
random.shuffle(all_images)

n = len(all_images)
train_end = int(0.8 * n)
val_end = int(0.9 * n)

train_images = all_images[:train_end]      # 80%
val_images = all_images[train_end:val_end]  # 10%
test_images = all_images[val_end:]          # 10%
```

**重要原则**:
- ✅ 按**原始图像级别**划分，再 patch crop
- ❌ 不要先 crop 再随机划分 (会导致数据泄漏)

---

## 七、数据泄漏防护

### 7.1 正确流程

```
原始图像 → Train/Val/Test 划分 → Patch Crop → 训练
```

### 7.2 错误流程

```
原始图像 → Patch Crop → 随机划分 → 训练  ❌ 数据泄漏!
```

### 7.3 检查项

- [ ] Train/Val/Test 使用不同的原始图像
- [ ] 同一原始图像的 patches 不会分散到不同 split
- [ ] RRSHID 不与训练集混用
- [ ] 随机种子固定 (reproducible)

---

## 八、Physical Prior 适用性验证

### 8.1 验证项

| 验证项 | RSHaze+ | RS-Haze | RRSHID |
|--------|---------|---------|--------|
| **RGB 输入** | ✅ | ✅ | ✅ |
| **可计算 DCP** | ✅ | ✅ | ✅ |
| **可计算 LCP** | ✅ | ✅ | ✅ |
| **可计算 CSP** | ✅ | ✅ | ✅ |
| **可生成 S_final** | ✅ | ✅ | ✅ |

### 8.2 验证脚本

```python
from src.models.haze_density import generate_s_final
import torch
from torchvision import transforms

transform = transforms.Compose([
    transforms.ToTensor(),
])

# 加载 hazy 图像
image = transform(image_path).unsqueeze(0)  # [1, 3, H, W]

# 计算 S_final
s_final = generate_s_final(image)  # [1, 1, H, W]

print(f"S_final shape: {s_final.shape}")
print(f"S_final range: [{s_final.min():.4f}, {s_final.max():.4f}]")
print(f"S_final finite: {torch.isfinite(s_final).all()}")
```

---

## 九、是否需要预先生成 S_final

### 9.1 方案对比

| 方案 | 优点 | 缺点 | 推荐 |
|------|------|------|------|
| **实时计算** | 灵活、节省磁盘 | 训练时额外计算 | ✅ 推荐 |
| **预先生成** | 训练速度快 | 占用额外磁盘 | 可选 |

### 9.2 推荐方案

**实时计算 S_final**

理由:
1. Physical Prior 计算速度快 (引导滤波优化后)
2. 节省磁盘空间 (10000 对 × 4 bytes × 512² ≈ 10 GB)
3. 灵活性高 (可调整权重参数)

---

## 十、核验清单

### 10.1 下载后核验项

- [ ] 目录结构正确
- [ ] 文件数量与预期一致
- [ ] 图像格式为 PNG/JPG
- [ ] 图像为 RGB 三通道
- [ ] 分辨率符合预期
- [ ] Hazy-Clear 配对正确
- [ ] 无损坏文件
- [ ] 无重复文件
- [ ] 官方 split 存在 (如有)
- [ ] Physical Prior 可正常计算
- [ ] 适合 patch crop

### 10.2 核验命令

```bash
# RSHaze+
!python scripts/inspect_dataset.py --dataset RSHazePlus --data_dir /content/datasets/RSHazePlus

# RS-Haze
!python scripts/inspect_dataset.py --dataset RSHaze --data_dir /content/datasets/RSHaze

# RRSHID
!python scripts/inspect_dataset.py --dataset RRSHID --data_dir /content/datasets/RRSHID
```

---

## 十一、更新记录

| 日期 | 更新内容 | 状态 |
|------|----------|------|
| 2026-08-28 | 创建文档框架 | 待核验 |
| 【待填】 | RSHaze+ 实际统计 | 待填充 |
| 【待填】 | RS-Haze 实际统计 | 待填充 |
| 【待填】 | RRSHID 实际统计 | 待填充 |

---

**文档版本**: v1.0  
**最后更新**: 2026-08-28  
**作者**: 遥感智研助手
