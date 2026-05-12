# -*- coding: utf-8 -*-
"""
SRCNN 图像超分辨率推理脚本
功能：加载训练好的模型，对单张图片进行超分重建
输出：双立方插值结果 + SRCNN 超分结果
"""

import argparse

import torch
import torch.backends.cudnn as cudnn
import numpy as np
import PIL.Image as pil_image

# 导入模型与工具函数
from models import SRCNN
from utils import convert_rgb_to_ycbcr, convert_ycbcr_to_rgb, calc_psnr

if __name__ == '__main__':
    # ======================== 1. 命令行参数解析 ========================
    parser = argparse.ArgumentParser(description='SRCNN 单张图像超分推理')
    parser.add_argument('--weights-file', type=str, required=True, help='训练好的模型权重路径')
    parser.add_argument('--image-file', type=str, required=True, help='输入待超分的图片路径')
    parser.add_argument('--scale', type=int, default=2, help='超分倍数，默认 2 倍')
    args = parser.parse_args()

    # ======================== 2. 设备配置 ========================
    cudnn.benchmark = True
    # 自动选择 GPU 或 CPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # ======================== 3. 构建模型并加载权重 ========================
    model = SRCNN().to(device)

    # 加载模型权重（兼容 CPU/GPU）
    state_dict = model.state_dict()
    for n, p in torch.load(args.weights_file, map_location=lambda storage, loc: storage).items():
        if n in state_dict.keys():
            state_dict[n].copy_(p)
        else:
            raise KeyError(n)

    # 设置模型为评估模式（禁用 Dropout/BatchNorm）
    model.eval()

    # ======================== 4. 读取图像并预处理 ========================
    # 读取图像并转为 RGB 格式
    image = pil_image.open(args.image_file).convert('RGB')

    # 将图像尺寸对齐为 scale 的整数倍（避免尺寸不匹配）
    image_width = (image.width // args.scale) * args.scale
    image_height = (image.height // args.scale) * args.scale
    image = image.resize((image_width, image_height), resample=pil_image.BICUBIC)

    # 先下采样 → 再上采样，构造双立方插值基线图像
    image = image.resize((image.width // args.scale, image.height // args.scale), resample=pil_image.BICUBIC)
    image = image.resize((image.width * args.scale, image.height * args.scale), resample=pil_image.BICUBIC)

    # 保存双立方插值结果
    image.save(args.image_file.replace('.', '_bicubic_x{}.'.format(args.scale)))

    # ======================== 5. 颜色空间转换（RGB → YCbCr） ========================
    # 转为 numpy 并归一化
    image = np.array(image).astype(np.float32)
    # 转换为 YCbCr 色彩空间（超分只在 Y 通道进行）
    ycbcr = convert_rgb_to_ycbcr(image)

    # 提取 Y 通道（亮度）
    y = ycbcr[..., 0]
    y /= 255.0  # 归一化到 [0, 1]
    y = torch.from_numpy(y).to(device)
    y = y.unsqueeze(0).unsqueeze(0)  # 增加 batch 和 channel 维度

    # ======================== 6. 模型推理（无梯度模式） ========================
    with torch.no_grad():
        preds = model(y).clamp(0.0, 1.0)  # 输出限制在 [0,1]

    # ======================== 7. 计算 PSNR 指标 ========================
    psnr = calc_psnr(y, preds)
    print('PSNR: {:.2f}'.format(psnr))

    # ======================== 8. 后处理：重建 RGB 图像 ========================
    # 恢复到 0~255 范围并转为 numpy
    preds = preds.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)

    # 合并 Y（超分后）+ Cb + Cr 通道
    output = np.array([preds, ycbcr[..., 1], ycbcr[..., 2]]).transpose([1, 2, 0])
    # 转回 RGB 并截断到有效范围
    output = np.clip(convert_ycbcr_to_rgb(output), 0.0, 255.0).astype(np.uint8)
    # 转为 PIL 图像
    output = pil_image.fromarray(output)

    # 保存 SRCNN 超分结果
    output.save(args.image_file.replace('.', '_srcnn_x{}.'.format(args.scale)))