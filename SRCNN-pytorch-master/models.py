# -*- coding: utf-8 -*-
"""
SRCNN 模型定义
论文：Image Super-Resolution Using Deep Convolutional Networks
结构：3层卷积神经网络，实现端到端单图像超分辨率重建
"""

from torch import nn


class SRCNN(nn.Module):
    """
    SRCNN (Super-Resolution Convolutional Neural Network)
    经典图像超分辨率卷积神经网络模型
    """

    def __init__(self, num_channels=1):
        """
        初始化 SRCNN 模型结构
        :param num_channels: 输入图像通道数，默认 1（仅亮度通道 Y 通道）
        """
        super(SRCNN, self).__init__()

        # 第一层卷积：特征提取
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=9, padding=9 // 2)
        # 第二层卷积：特征映射
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=5 // 2)
        # 第三层卷积：高分辨率图像重建
        self.conv3 = nn.Conv2d(32, num_channels, kernel_size=5, padding=5 // 2)

        # 激活函数 ReLU
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        前向传播
        :param x: 输入低分辨率图像张量
        :return: 输出高分辨率图像张量
        """
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x