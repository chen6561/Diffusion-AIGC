# 🧨 DDPM 图像生成模型（完整可运行版）
**极简、稳定、开箱即用的 PyTorch DDPM 扩散模型实现**

[![](https://img.shields.io/badge/PyTorch-1.12+-orange.svg)](https://pytorch.org/)
[![](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org/)
[![](https://img.shields.io/badge/DDPM-Implementation-green.svg)](https://arxiv.org/abs/2006.11239)
[![](https://img.shields.io/badge/Windows-Linux-red.svg)]()

---

## 📖 项目介绍
本项目是 **完整可运行的 DDPM（Denoising Diffusion Probabilistic Models）** 图像生成实现，包含：
- 标准 UNet 主干网络
- 时间嵌入（Time Embedding）
- 前向扩散 + 反向采样
- 自定义数据集训练
- 自动保存模型 + 采样可视化

**一键训练 → 自动生成图像**

---

## 📁 项目结构
```
Diffusion-AIGC/
├── config.py          # 超参数统一配置
├── train.py           # 训练主脚本
├── ddpm.py            # DDPM 扩散核心逻辑
├── models.py          # UNet + 残差块 + 时间编码
├── datasets.py        # 自定义图像数据集加载
├── utils.py           # 模型保存 + 画图工具
├── requirements.txt   # 依赖环境
├── checkpoints/       # 模型权重保存
├── logs/              # 生成样本保存
└── data/train/        # 训练图片存放（你只需要动这里）
```

---

## ✨ 核心特性
✅ **完全可直接运行，无任何 Bug**
✅ **自定义数据集训练**（只需放入图片）
✅ **UNet + Time Embedding + ResBlock 标准结构**
✅ **MSE 噪声预测损失（DDPM 原始论文）**
✅ **自动保存 checkpoint + 自动生成样本图**
✅ **解决 Windows OpenMP 冲突**
✅ **全中文注释，适合学习与教学**
✅ **GPU / CPU 自动识别**

---

## 🚀 快速开始

### 1. 安装环境
```bash
pip install -r requirements.txt
```

### 2. 准备数据集
把你的训练图片放入：
```
data/train/xxx.jpg
data/train/xxx.png
...
```

### 3. 启动训练
```bash
python train.py
```

### 4. 查看结果
- 模型权重 → `checkpoints/`
- 生成图片 → `logs/samples_epoch_x.png`

---

## ⚙️ 核心参数（config.py）
```python
image_size = 128        # 图像尺寸
batch_size = 32         # 批次大小
base_channels = 128     # 模型宽度
num_timesteps = 1000    # 扩散步数
lr = 2e-4               # 学习率
epochs = 100            # 训练轮数
save_every = 5          # 每5轮保存一次
```

---

## 📈 训练流程
1. 图像归一化 → [-1, 1]
2. 前向扩散：逐步加噪
3. UNet 预测噪声
4. MSE 损失反向传播
5. 保存模型
6. 从纯噪声生成清晰图像

---

## 🧪 推理采样（生成新图）
```python
import torch
from ddpm import DDPM
from config import DDPMConfig

config = DDPMConfig()
ddpm = DDPM(config)

# 加载训练好的模型
ckpt = torch.load("checkpoints/ddpm_epoch_50.pth")
ddpm.load_state_dict(ckpt["model_state_dict"])

# 生成 4 张图
ddpm.eval()
with torch.no_grad():
    samples = ddpm.p_sample_loop(batch_size=4)
```

---

## 🛠 常见问题
### 1. OMP: Error #15 冲突
已内置修复，**直接运行即可**。

### 2. CUDA 显存不足
调小 `batch_size` 或 `image_size`。

### 3. 训练速度慢
建议使用 **Nvidia GPU + CUDA**。

---

## 📄 许可证
MIT License  
可自由用于 **学习、科研、二次开发**。

---

## 📩 作者
完整注释版 DDPM 项目  
**一键训练 · 开箱即用 · 稳定无 Bug**

---
