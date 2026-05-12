# Diffusion-AIGC：基于扩散模型的AIGC生成与算法研究平台
**面向高级算法工程师 | 论文复现 | 工业落地 | 生成式AI工程化**

---

## 项目定位
本项目是**Diffusion 扩散模型 + AIGC 内容生成**的一站式研究与工程化平台，专注于：
- 主流扩散模型（Stable Diffusion / DDPM / LDM / DiT / ControlNet / DiffBIR）复现与自研
- 文生图、图生图、超分辨率、瑕疵修复、图像编辑等 AIGC 能力
- 工业场景落地（如你正在做的：电池缺陷检测、视觉质量增强、AI 生成式质检）
- 训练、推理、加速、部署全流程工具链

适合：**算法学习、论文复现、工业AIGC项目快速搭建**

---

## ✨ 核心功能
### 🧨 Diffusion 模型能力
- 标准 DDPM 正向加噪 / 反向去噪流程实现
- Latent Diffusion（LDM）潜在空间扩散
- DiT（Diffusion Transformer）基于Transformer的扩散架构
- ControlNet 条件控制生成
- DiffBIR 盲图像复原
- 图像超分、瑕疵修复、缺陷增强

### 🎨 AIGC 生成能力
- 文本引导图像生成
- 参考图风格/结构迁移
- 工业缺陷图像仿真生成
- 低质图像修复与高清化
- 自定义数据集训练与微调

### 🔧 工程化能力
- 统一训练/推理接口
- 支持多GPU、分布式、混合精度训练
- 模型导出（ONNX/TensorRT）
- 实验配置化管理
- 结果自动保存与可视化

---

## 🧱 项目结构
```
Diffusion-AIGC/
├── configs/                # 训练/推理配置文件
├── core/                   # 核心模块
│   ├── diffusion/          # DDPM、LDM、DiT 扩散实现
│   ├── models/             # UNet、Transformer、ControlNet
│   ├── modules/            # 基础组件：Attention、EMA、归一化
│   └── sampler/            # 采样器：DDIM、DPM-Solver
├── datasets/               # 数据集加载与处理
├── tools/                  # 训练、推理、评估脚本
├── utils/                  # 工具函数
├── assets/                 # 效果图、结构示意图
├── train.py                # 训练入口
├── inference.py            # 推理/生成入口
└── README.md
```

---

## 🚀 快速开始
### 1. 克隆项目
```bash
git clone https://github.com/xxx/Diffusion-AIGC.git
cd Diffusion-AIGC
```

### 2. 安装环境
```bash
pip install torch torchvision
pip install -r requirements.txt
```

### 3. 快速推理（DiffBIR 示例）
```bash
python inference.py \
    --config configs/diffbir.yaml \
    --input assets/test/ \
    --output results/ \
    --ckpt weights/diffbir_final.pth
```

### 4. 训练 Diffusion 模型
```bash
python train.py --config configs/ddpm.yaml
```

---

## 📌 已支持/计划支持模型
✅ **Diffusion 基础模型**
- DDPM
- Latent Diffusion (LDM)
- DiT (Diffusion Transformer)
- DDIM, DPM-Solver 加速采样

✅ **AIGC 实用模型**
- Stable Diffusion 系列
- ControlNet
- **DiffBIR（盲图像复原）**
- 图像超分 / 瑕疵修复 / 缺陷生成

---

## 🎯 适合场景
- 工业视觉缺陷检测（仿真样本生成、瑕疵修复）
- 图像质量增强、超分辨率
- AIGC 文生图/图生图
- 算法研究、论文复现
- 大厂面试项目展示

---

## 📈 技术栈
PyTorch | Diffusion | Transformer | AIGC | CV |
CUDA | Distributed Training | ONNX | TensorRT |

---

## 📩 说明
本项目用于**个人研究与论文复现**，所有代码遵循学术开源协议。
**不包含任何公司涉密数据、业务逻辑或私有权重。**

---
