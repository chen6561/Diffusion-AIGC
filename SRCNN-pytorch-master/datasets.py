# -*- coding: utf-8 -*-
"""
SRCNN 数据集加载模块
功能：定义训练集和验证集的数据读取类
基于 h5 格式数据集，返回低分辨率（lr）和高分辨率（hr）图像对
"""

import h5py
import numpy as np
from torch.utils.data import Dataset


class TrainDataset(Dataset):
    """
    训练数据集类
    加载 h5 格式的训练数据，返回归一化后的 [lr, hr] 图像对
    """
    def __init__(self, h5_file):
        """
        初始化数据集
        :param h5_file: 训练集 .h5 文件路径
        """
        super(TrainDataset, self).__init__()
        self.h5_file = h5_file  # 保存 h5 文件路径

    def __getitem__(self, idx):
        """
        根据索引获取单张训练数据
        :param idx: 数据索引
        :return: lr 低分辨率图像, hr 高分辨率图像
        """
        with h5py.File(self.h5_file, 'r') as f:
            # 读取数据并归一化到 [0,1]，增加通道维度 (C, H, W)
            lr = np.expand_dims(f['lr'][idx] / 255., 0)
            hr = np.expand_dims(f['hr'][idx] / 255., 0)
            return lr, hr

    def __len__(self):
        """返回数据集总长度"""
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['lr'])


class EvalDataset(Dataset):
    """
    验证/测试数据集类
    加载 h5 格式的验证数据，返回归一化后的 [lr, hr] 图像对
    与训练集不同：验证集通过字符串索引读取
    """
    def __init__(self, h5_file):
        """
        初始化验证数据集
        :param h5_file: 验证集 .h5 文件路径
        """
        super(EvalDataset, self).__init__()
        self.h5_file = h5_file

    def __getitem__(self, idx):
        """
        根据索引获取单张验证数据
        :param idx: 数据索引
        :return: lr 低分辨率图像, hr 高分辨率图像
        """
        with h5py.File(self.h5_file, 'r') as f:
            # 验证集使用字符串索引读取整张图像
            lr = np.expand_dims(f['lr'][str(idx)][:, :] / 255., 0)
            hr = np.expand_dims(f['hr'][str(idx)][:, :] / 255., 0)
            return lr, hr

    def __len__(self):
        """返回验证集总长度"""
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['lr'])