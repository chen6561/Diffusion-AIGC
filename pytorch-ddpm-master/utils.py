# 操作系统相关工具，用于创建文件夹、路径拼接
import os
# PyTorch核心库
import torch
# 绘图库，用于生成采样图片
import matplotlib.pyplot as plt
# 导入项目配置类
from config import DDPMConfig


def save_checkpoint(config: DDPMConfig, model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int):
    """
    保存模型检查点（用于断点续训 + 推理）
    Args:
        config: 配置对象
        model: DDPM模型
        optimizer: 优化器
        epoch: 当前训练轮数
    """
    # 创建模型保存文件夹（如果不存在则创建，存在不报错）
    os.makedirs(config.save_dir, exist_ok=True)
    # 拼接检查点保存路径
    checkpoint_path = os.path.join(config.save_dir, f"ddpm_epoch_{epoch}.pth")

    # 保存内容：轮数、模型权重、优化器权重、配置文件
    torch.save({
        "epoch": epoch,  # 当前epoch
        "model_state_dict": model.state_dict(),  # 模型权重
        "optimizer_state_dict": optimizer.state_dict(),  # 优化器状态
        "config": config  # 完整配置
    }, checkpoint_path)

    # 打印保存信息
    print(f"Checkpoint saved to {checkpoint_path}")


def denormalize_image(img: torch.Tensor) -> torch.Tensor:
    """
    反归一化：将 [-1, 1] 范围的图像恢复到 [0, 1]，方便可视化
    因为训练时通常做了 normalize = (img - 0.5)/0.5 → 转为 [-1,1]
    """
    return (img * 0.5 + 0.5).clamp(0, 1)


def plot_samples(samples: torch.Tensor, save_path: str = None):
    """
    可视化模型生成的图片，并可选择保存
    Args:
        samples: 模型生成的图像张量 [B, C, H, W]
        save_path: 保存路径（不传则只显示不保存）
    """
    # 1. 将图像移到CPU，并反归一化到 [0,1]
    samples = denormalize_image(samples.cpu())

    # 2. 获取生成图片的数量
    n = samples.shape[0]

    # 3. 创建画布
    plt.figure(figsize=(10, 10))

    # 4. 逐张绘制图片
    for i in range(n):
        # 子图：1行n列
        plt.subplot(1, n, i + 1)
        # 将 [C, H, W] → [H, W, C] 以适应matplotlib格式
        plt.imshow(samples[i].permute(1, 2, 0))
        # 关闭坐标轴
        plt.axis("off")

    # 5. 如果传入保存路径，则保存图片
    if save_path:
        # 先创建文件夹
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # 保存图片，去除白边
        plt.savefig(save_path, bbox_inches="tight")

    # 关闭画布，释放内存
    plt.close()