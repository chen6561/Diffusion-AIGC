# 解决Windows下OpenMP重复加载冲突（必须放在导入torch之前）
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 深度学习框架核心库
import torch
# 优化器：AdamW 用于模型参数更新
import torch.optim as optim
# 进度条工具，用于训练可视化
from tqdm import tqdm

# 导入项目自定义模块
from config import DDPMConfig        # 模型/训练配置类
from datasets import get_dataloader  # 数据集加载函数
from ddpm import DDPM                # DDPM扩散模型核心类
from utils import save_checkpoint, plot_samples  # 工具函数：保存模型、画图


def train(config: DDPMConfig):
    """
    DDPM 模型训练主函数
    参数：config - 包含所有超参数的配置对象
    """
    # 1. 初始化训练数据加载器
    train_loader = get_dataloader(config, train=True)

    # 2. 初始化 DDPM 模型与 AdamW 优化器
    ddpm = DDPM(config)
    optimizer = optim.AdamW(ddpm.model.parameters(), lr=config.lr, weight_decay=1e-4)

    # 3. 开始训练循环
    ddpm.train()  # 将模型设置为训练模式（启用Dropout/BatchNorm）
    for epoch in range(config.epochs):
        total_loss = 0.0  # 累计本轮总损失
        # 构建本轮训练的进度条
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.epochs}")

        # 遍历每一个批次数据
        for batch in pbar:
            # 将数据移动到指定设备（GPU/CPU）
            batch = batch.to(config.device)

            # 随机采样时间步 t，形状与 batch 一致
            t = torch.randint(
                0, config.num_timesteps, (batch.shape[0],),
                device=config.device, dtype=torch.long
            )

            # 前向传播：计算当前批次的噪声预测损失
            loss = ddpm.loss(batch, t)

            # 反向传播：更新模型参数
            optimizer.zero_grad()    # 清空上一步梯度
            loss.backward()          # 梯度回传
            torch.nn.utils.clip_grad_norm_(ddpm.model.parameters(), 1.0)  # 梯度裁剪，防止爆炸
            optimizer.step()         # 执行一步参数更新

            # 更新训练日志与进度条显示
            total_loss += loss.item()
            pbar.set_postfix({"loss": loss.item(), "avg_loss": total_loss / (pbar.n + 1)})

        # 4. 每隔指定轮数保存模型并生成采样图片
        if (epoch + 1) % config.save_every == 0:
            # 保存模型权重与优化器状态
            save_checkpoint(config, ddpm, optimizer, epoch + 1)

            # 切换到评估模式，禁止梯度计算
            ddpm.eval()
            # 从纯噪声中采样生成图片
            samples = ddpm.p_sample_loop(batch_size=4)
            # 保存采样结果到日志文件夹
            plot_samples(samples, save_path=f"{config.log_dir}/samples_epoch_{epoch + 1}.png")
            # 切回训练模式
            ddpm.train()


if __name__ == "__main__":
    # 程序入口：加载配置并启动训练
    config = DDPMConfig()
    train(config)