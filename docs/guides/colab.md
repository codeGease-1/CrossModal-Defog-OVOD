# Google Colab 使用说明

本文档说明如何在 Google Colab 免费 T4 GPU 上运行本项目。

---

## 一、准备工作

### 1. Google 账号

确保你有一个 Google 账号，用于访问 Colab 和 Google Drive。

### 2. GitHub 账号（可选）

如果项目已推送到 GitHub，可以直接 clone；否则需要上传 ZIP 文件。

---

## 二、Colab 初始化步骤

### 方式 A：从 GitHub Clone（推荐）

```python
# ========================
# Step 1: 挂载 Google Drive
# ========================
from google.colab import drive
drive.mount('/content/drive')

# ========================
# Step 2: 克隆项目
# ========================
# 替换为你的 GitHub 仓库地址
!git clone https://github.com/your-username/CrossModal-Defog-OVOD.git

# 或者从 Google Drive 复制（如果已上传）
# !cp -r /content/drive/MyDrive/CrossModal-Defog-OVOD /content/

# ========================
# Step 3: 进入项目目录
# ========================
%cd /content/CrossModal-Defog-OVOD

# ========================
# Step 4: 安装依赖
# ========================
!pip install -r requirements.txt

# ========================
# Step 5: 设置 PYTHONPATH
# ========================
import sys
sys.path.insert(0, '/content/CrossModal-Defog-OVOD')

# ========================
# Step 6: 检查 GPU 环境
# ========================
!nvidia-smi

# 检查 PyTorch GPU 支持
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB" 
      if torch.cuda.is_available() else "No GPU")
```

---

### 方式 B：上传 ZIP 文件

```python
# ========================
# Step 1: 挂载 Google Drive
# ========================
from google.colab import drive
drive.mount('/content/drive')

# ========================
# Step 2: 上传 ZIP 文件
# ========================
from google.colab import files
uploaded = files.upload()

# ========================
# Step 3: 解压
# ========================
!unzip CrossModal-Defog-OVOD.zip
%cd CrossModal-Defog-OVOD

# ========================
# Step 4: 安装依赖
# ========================
!pip install -r requirements.txt

# ========================
# Step 5: 设置 PYTHONPATH
# ========================
import sys
sys.path.insert(0, '/content/CrossModal-Defog-OVOD')
```

---

## 三、运行训练

### 基本训练

```python
# 使用默认配置训练
!python scripts/train_haze_density.py --config configs/haze_density.yaml
```

### 自定义参数训练

```python
# 修改 batch_size 和 learning_rate
!python scripts/train_haze_density.py \
    --config configs/haze_density.yaml \
    --data.batch_size 8 \
    --train.lr 5e-4
```

### 断点续训

```python
# 从 checkpoint 继续训练
!python scripts/train_haze_density.py \
    --config configs/haze_density.yaml \
    --resume experiments/haze_density/checkpoints/latest.pt
```

---

## 四、保存实验结果

### 自动保存

训练过程中，checkpoint 会自动保存到：
- `experiments/haze_density/checkpoints/` - 模型权重
- `experiments/haze_density/logs/` - TensorBoard 日志

### 同步到 Google Drive

```python
# 训练完成后，同步到 Google Drive
!cp -r /content/CrossModal-Defog-OVOD/experiments /content/drive/MyDrive/CrossModal-Defog-OVOD-experiments/
```

### 下载 Checkpoint

```python
from google.colab import files

# 压缩 checkpoints
!tar -czf checkpoints.tar.gz experiments/haze_density/checkpoints/

# 下载
files.download('checkpoints.tar.gz')
```

---

## 五、TensorBoard 可视化

```python
# 启动 TensorBoard
%load_ext tensorboard
%tensorboard --logdir experiments/haze_density/logs
```

或在 Colab 菜单中选择：Tools → TensorBoard

---

## 六、GPU 资源管理

### T4 GPU 显存限制

- **总显存**: ~16 GB
- **推荐 batch_size**: 4-8 (image_size=256)
- **最大 batch_size**: 视模型复杂度而定

### 显存监控

```python
import torch

print(f"Allocated memory: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
print(f"Reserved memory: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
```

### 显存清理

```python
import torch
import gc

torch.cuda.empty_cache()
gc.collect()
```

---

## 七、常见问题

### Q1: CUDA out of memory

**解决**: 减小 batch_size

```yaml
# 在 configs/haze_density.yaml 中修改
data:
  batch_size: 2  # 从 4 降到 2
```

### Q2: 依赖安装失败

**解决**: 重新运行安装命令，或检查网络连接

```bash
!pip install -r requirements.txt --upgrade
```

### Q3: Colab 会话断开

**解决**: 
- Checkpoint 已自动保存，重新挂载 Drive 后继续
- 使用断点续训命令恢复训练

### Q4: GPU 超时释放

**解决**: 
- Colab 免费 tier 最长运行约 12 小时
- 定期保存 checkpoint
- 考虑使用 Colab Pro 或本地 GPU

---

## 八、脚本说明

### `scripts/setup_colab.py`

一键初始化 Colab 环境：

```python
# 在 Colab 中运行
!python scripts/setup_colab.py
```

该脚本自动执行：
1. 检查 GPU 可用性
2. 验证 PyTorch 版本
3. 打印环境信息
4. 创建必要的输出目录

---

## 九、下一步

完成初始化后，运行以下命令进行 smoke test：

```python
# 形状测试（不训练，仅验证 forward/backward）
!python scripts/smoke_test.py
```

---

**最后更新**: 2026-08-28
