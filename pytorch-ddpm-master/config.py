# 导入PyTorch核心库
import torch
# 导入数据类装饰器，用于简洁定义配置类
from dataclasses import dataclass
# 导入类型注解工具，用于指定元组类型
from typing import Tuple


@dataclass
class DDPMConfig:
    """DDPM模型配置类：统一管理数据集、模型结构、训练策略、保存日志等所有超参数"""

    # ====================== 数据集加载相关参数 ======================
    # 数据集存放的根路径
    dataset_path: str = "./data"
    # 输入图像统一缩放为该尺寸（正方形）
    image_size: int = 128
    # 训练时每个批次的样本数量
    batch_size: int = 32
    # 数据加载器使用的线程数
    num_workers: int = 4

    # ====================== UNet模型结构参数 ======================
    # 输入图像通道数（RGB为3，灰度图为1）
    in_channels: int = 3
    # UNet基础通道数，控制模型宽度
    base_channels: int = 128
    # 通道倍数，控制下采样/上采样各阶段通道变化
    channel_mults: Tuple[int, ...] = (1, 2, 4, 8)
    # 每个尺度下使用的残差块数量
    num_res_blocks: int = 2
    # Dropout失活概率，用于防止过拟合
    dropout: float = 0.1
    # 时间步嵌入向量的维度
    time_emb_dim = 256

    # ====================== 训练过程超参数 ======================
    # 优化器学习率
    lr: float = 2e-4
    # 扩散过程β的初始值
    beta_start: float = 1e-4
    # 扩散过程β的最终值
    beta_end: float = 0.02
    # 扩散模型总时间步T
    num_timesteps: int = 1000
    # 训练总轮数
    epochs: int = 100
    # 运行设备：自动检测是否使用GPU
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # ====================== 日志与模型保存参数 ======================
    # 日志、采样图片保存目录
    log_dir: str = "./logs"
    # 模型权重文件保存目录
    save_dir: str = "./checkpoints"
    # 每隔多少轮保存一次模型
    save_every: int = 5