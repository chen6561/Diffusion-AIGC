# -*- coding: utf-8 -*-
"""
图像预处理工具库
包含 RGB <-> YCbCr 颜色空间转换、PSNR计算、平均指标计算器

功能：
1. RGB 转 Y 通道（超分模型只训练 Y 通道）
2. RGB ↔ YCbCr 互转
3. 计算 PSNR 指标
4. AverageMeter 用于训练时统计平均损失/指标
"""

import torch
import numpy as np


def convert_rgb_to_y(img):
    """
    将 RGB 图像转换为 Y 通道（亮度通道）
    超分辨率模型通常只在 Y 通道上训练，效果更好

    参数:
        img: np.ndarray (H, W, 3) 或 torch.Tensor (3, H, W)
    返回:
        y: 亮度通道单通道图像
    """
    if isinstance(img, np.ndarray):
        # BT.601 标准 RGB 转 Y
        return 16. + (64.738 * img[:, :, 0] + 129.057 * img[:, :, 1] + 25.064 * img[:, :, 2]) / 256.

    elif isinstance(img, torch.Tensor):
        # 如果是 batch (1, 3, H, W)，去掉 batch 维度
        if len(img.shape) == 4:
            img = img.squeeze(0)
        # RGB 张量格式 (3, H, W)
        return 16. + (64.738 * img[0, :, :] + 129.057 * img[1, :, :] + 25.064 * img[2, :, :]) / 256.

    else:
        raise Exception('不支持的图像类型:', type(img))


def convert_rgb_to_ycbcr(img):
    """
    RGB 转换为 YCbCr 颜色空间（BT.601 标准）
    支持 numpy array 和 torch tensor

    参数:
        img: RGB 图像 (HWC 格式 np / CHW 格式 tensor)
    返回:
        ycbcr: 转换后的 YCbCr 图像
    """
    if isinstance(img, np.ndarray):
        # RGB 转 YCbCr 公式
        y = 16. + (64.738 * img[:, :, 0] + 129.057 * img[:, :, 1] + 25.064 * img[:, :, 2]) / 256.
        cb = 128. + (-37.945 * img[:, :, 0] - 74.494 * img[:, :, 1] + 112.439 * img[:, :, 2]) / 256.
        cr = 128. + (112.439 * img[:, :, 0] - 94.154 * img[:, :, 1] - 18.285 * img[:, :, 2]) / 256.
        # 拼接并转为 HWC 格式
        return np.array([y, cb, cr]).transpose([1, 2, 0])

    elif isinstance(img, torch.Tensor):
        if len(img.shape) == 4:
            img = img.squeeze(0)
        # RGB 张量格式 (3, H, W)
        y = 16. + (64.738 * img[0, :, :] + 129.057 * img[1, :, :] + 25.064 * img[2, :, :]) / 256.
        cb = 128. + (-37.945 * img[0, :, :] - 74.494 * img[1, :, :] + 112.439 * img[2, :, :]) / 256.
        cr = 128. + (112.439 * img[0, :, :] - 94.154 * img[1, :, :] - 18.285 * img[2, :, :]) / 256.

        # 通道拼接 (3, H, W)
        ycrcb = torch.cat([y.unsqueeze(0), cb.unsqueeze(0), cr.unsqueeze(0)], dim=0)
        return ycrcb.permute(1, 2, 0)  # 转为 (H, W, 3)

    else:
        raise Exception('不支持的图像类型:', type(img))


def convert_ycbcr_to_rgb(img):
    """
    YCbCr 转换回 RGB 颜色空间
    用于超分后重建彩色图像
    """
    if isinstance(img, np.ndarray):
        # YCbCr 转回 RGB 公式
        r = 298.082 * img[:, :, 0] / 256. + 408.583 * img[:, :, 2] / 256. - 222.921
        g = 298.082 * img[:, :, 0] / 256. - 100.291 * img[:, :, 1] / 256. - 208.120 * img[:, :, 2] / 256. + 135.576
        b = 298.082 * img[:, :, 0] / 256. + 516.412 * img[:, :, 1] / 256. - 276.836
        return np.array([r, g, b]).transpose([1, 2, 0])

    elif isinstance(img, torch.Tensor):
        if len(img.shape) == 4:
            img = img.squeeze(0)
        # 张量格式转换
        r = 298.082 * img[0, :, :] / 256. + 408.583 * img[2, :, :] / 256. - 222.921
        g = 298.082 * img[0, :, :] / 256. - 100.291 * img[1, :, :] / 256. - 208.120 * img[2, :, :] / 256. + 135.576
        b = 298.082 * img[0, :, :] / 256. + 516.412 * img[1, :, :] / 256. - 276.836
        return torch.cat([r.unsqueeze(0), g.unsqueeze(0), b.unsqueeze(0)], dim=0).permute(1, 2, 0)

    else:
        raise Exception('不支持的图像类型:', type(img))


def calc_psnr(img1, img2):
    """
    计算两张图像的 PSNR 峰值信噪比
    用于评估超分模型效果，值越高越清晰

    参数:
        img1: 预测图像
        img2: 真实图像
    返回:
        psnr: 标量指标
    """
    # 图像归一化到 [0,1]，MSE 计算
    return 10. * torch.log10(1. / torch.mean((img1 - img2) ** 2))


class AverageMeter(object):
    """
    用于计算并保存平均值（训练时的损失、指标统计）
    例如：平均 loss、平均 PSNR
    """

    def __init__(self):
        # 初始化时重置所有参数
        self.reset()

    def reset(self):
        """重置指标"""
        self.val = 0  # 当前值
        self.avg = 0  # 平均值
        self.sum = 0  # 累加和
        self.count = 0  # 累计数量

    def update(self, val, n=1):
        """
        更新指标
        参数:
            val: 当前值
            n: 样本数量（batch size）
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count