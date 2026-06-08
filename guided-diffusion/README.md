# Guided-Diffusion Super Resolution
基于 **Guided-Diffusion** 实现的图像超分辨率扩散模型，包含**完整训练脚本**与**批量推理脚本**，开箱即用，适配单卡 GPU 训练/推理。

## 项目简介
本项目依托 OpenAI `guided-diffusion` 框架搭建超分辨率任务：
1. **训练阶段**：仅需提供高分辨率(HR)图像，代码自动下采样生成低分辨率(LR)图像，无需额外制作配对数据集；
2. **推理阶段**：输入低分辨率图像，通过扩散模型还原高清图像，支持批量处理；
3. 原生支持单卡 GPU、半精度加速、EMA 权重、断点续训、验证集可视化对比图；
4. 代码已做路径适配，自动解决 `guided_diffusion` 模块导入问题。

## 项目结构
```
guided-diffusion/
├── guided_diffusion/          # 官方核心库文件
├── scripts/
│   ├── super_res_train.py     # 超分辨率模型训练脚本
│   └── super_res_inference.py # 超分辨率图像推理脚本
├── dataset/                   # 自定义数据集目录（建议）
│   ├── train/HR/              # 训练集：高分辨率图像
│   └── val/HR/                # 验证集：高分辨率图像
└── results/                   # 训练输出：模型权重、日志、可视化结果
```

## 环境依赖
建议使用 Conda 搭建虚拟环境，Python 版本推荐 `3.8 ~ 3.10`，PyTorch >= 1.12。

### 安装依赖
```bash
# 基础依赖
pip install torch torchvision
pip install opencv-python numpy pillow

# 若缺少分布式/日志相关依赖，补充安装
pip install blobfile mpi4py
```

> 说明：本项目基于官方 `guided-diffusion` 二次开发，需保证项目根目录为 `guided-diffusion`。

## 1. 模型训练
### 1.1 数据集准备
仅准备**高分辨率图像**即可，代码自动生成低分辨率图像：
- 训练集：`dataset/train/HR/` 放入全部训练高清图
- 验证集：`dataset/val/HR/` 放入少量验证高清图（用于可视化效果）

### 1.2 核心参数配置
打开 `scripts/super_res_train.py`，修改底部默认参数（按需调整）：
```python
defaults = {
    "data_dir": "C:/dataset/train/HR",    # 训练集HR路径
    "val_dir": "C:/dataset/val/HR",       # 验证集HR路径
    "batch_size": 2,                      # 训练批次
    "lr": 1e-4,                           # 学习率
    "small_size": 64,                     # 低分辨率尺寸
    "large_size": 256,                    # 高分辨率尺寸
    "save_interval": 10,                  # 模型保存间隔(step)
    "resume_checkpoint": "",              # 断点续训权重路径，空=从头训练
    "use_fp16": False,                    # 半精度训练（显存不足开启）
}
```

### 1.3 启动训练
进入项目根目录 `guided-diffusion`，执行命令：
```bash
python scripts/super_res_train.py
```

### 训练特性
- 自动 HR → LR 下采样，省去数据集配对工作；
- 每轮保存权重时，自动输出 **LR / 超分结果 / 原图 HR** 拼接对比图；
- 支持 EMA 权重平滑、学习率衰减、权重正则化；
- 内置路径适配，自动加载 `guided_diffusion` 模块，无导入报错；
- 支持断点续训，填入 `resume_checkpoint` 权重路径即可恢复训练。

## 2. 图像推理（超分）
### 2.1 准备输入图片
将需要超分的**低分辨率图片**放入默认目录：`lr_images/`
支持格式：`jpg / jpeg / png / bmp`

### 2.2 推理参数配置
打开 `scripts/super_res_inference.py`，修改核心配置：
```python
defaults = dict(
    img_dir="./lr_images",                # 输入LR图像目录
    save_dir="./sr_results",              # 输出超分图像目录
    model_path="./results/model179000.pt",# 训练好的模型权重路径
    small_size = 128,                     # 输入LR尺寸（必须与训练一致）
    large_size = 512                      # 输出HR尺寸（必须与训练一致）
    use_fp16=False,                       # 半精度推理加速
    use_ddim=False                        # DDIM 快速采样（提速）
)
```

> ⚠️ **重要**：`small_size` / `large_size` 必须和训练脚本完全一致，否则模型加载/前向会报错。

### 2.3 启动推理
```bash
python scripts/super_res_inference.py
```

### 推理输出
- 输出图片自动命名为 `sr_原文件名`，统一保存至 `sr_results/`；
- 批量遍历文件夹内所有图片，全自动处理；
- 兼容多种权重格式，适配官方/自定义训练权重；
- 自动处理 RGB / BGR 格式转换，色彩正常无偏移。

## 关键注意事项
1. **尺寸一致性**
   训练与推理的 `small_size`、`large_size` 必须严格匹配，推荐标准倍率：`64→256`、`128→512`（4倍超分）。

2. **显存优化方案**
   - 降低 `batch_size`；
   - 开启 `use_fp16=True` 启用半精度训练/推理；
   - 训练时设置 `microbatch` 拆分批次，缓解显存压力。

3. **模块导入问题**
   脚本内部已自动追加项目根目录到 `sys.path`，正常运行不会出现 `ModuleNotFoundError: No module named 'guided_diffusion'`。
   若仍报错，手动添加环境变量：
   ```bash
   # Linux / Mac
   export PYTHONPATH=$PYTHONPATH:/path/to/guided-diffusion
   # Windows CMD
   set PYTHONPATH=%PYTHONPATH%;D:\work\git\Diffusion-AIGC\guided-diffusion
   ```

4. **断点续训**
   在训练脚本 `resume_checkpoint` 填入历史权重路径，即可从上一轮继续训练。

## 常见问题 FAQ
### Q1: 加载权重提示 `Missing key / Unexpected key`
A1: 模型结构参数不匹配，检查推理脚本的 `num_channels`、`num_res_blocks`、`attention_resolutions` 等结构参数，**与训练脚本保持一致**。

### Q2: 推理后图片颜色异常、偏色
A2: 脚本已内置 BGR ↔ RGB 转换，若仍异常，检查原始图片是否为非常规编码格式。

### Q3: 训练效果不佳、超分模糊
A3:
- 扩充训练数据集，保证高清图质量；
- 延长训练步数，多尝试不同 `save_interval` 产出的权重；
- 微调学习率、批次大小，优先使用 4 倍标准缩放比例。

## 许可证
本项目基于 OpenAI [guided-diffusion](https://github.com/openai/guided-diffusion) 开源框架二次开发，遵循原项目开源协议。