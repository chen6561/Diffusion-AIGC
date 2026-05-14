import os
import torch
import torch.distributed as dist
from tqdm import tqdm

from utils.utils import get_lr, show_result


def fit_one_epoch(
        diffusion_model_train: torch.nn.Module,
        diffusion_model: torch.nn.Module,
        loss_history,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        epoch_step: int,
        gen,
        Epoch: int,
        cuda: bool,
        fp16: bool,
        scaler,
        save_period: int,
        save_dir: str,
        local_rank: int = 0
):
    """
    扩散模型（DDPM）单轮训练核心函数
    包含：前向传播、损失计算、反向传播、参数更新、EMA更新、日志打印、模型保存

    Args:
        diffusion_model_train: 训练模式的扩散模型（用于计算损失）
        diffusion_model: 完整扩散模型（包含ema模型，用于推理和更新）
        loss_history: 损失记录工具类
        optimizer: 优化器
        epoch: 当前轮次
        epoch_step: 每轮迭代步数
        gen: 训练数据生成器
        Epoch: 总训练轮数
        cuda: 是否使用GPU
        fp16: 是否开启混合精度训练
        scaler: 混合精度梯度缩放器
        save_period: 每隔多少轮保存一次模型
        save_dir: 模型保存路径
        local_rank: 分布式训练进程编号（单卡默认为0）
    """
    # 初始化单轮总损失
    total_loss = 0.0

    # 主进程（local_rank=0）打印训练开始信息 + 初始化进度条
    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(
            total=epoch_step,
            desc=f'Epoch {epoch + 1}/{Epoch}',
            mininterval=0.3
        )

    # 遍历本轮所有批次数据
    for iteration, images in enumerate(gen):
        # 达到设定步数后停止
        if iteration >= epoch_step:
            break

        # 数据迁移至GPU，关闭梯度计算（仅数据搬运，无参数更新）
        with torch.no_grad():
            if cuda:
                images = images.cuda(local_rank)

        # -------------------#
        # 1. 正常精度训练
        # -------------------#
        if not fp16:
            # 清空上一步梯度
            optimizer.zero_grad()
            # 前向传播：计算扩散损失（模型预测噪声）
            diffusion_loss = torch.mean(diffusion_model_train(images))
            # 反向传播：计算梯度
            diffusion_loss.backward()
            # 优化器更新参数
            optimizer.step()

        # -------------------#
        # 2. FP16 混合精度训练
        # -------------------#
        else:
            from torch.cuda.amp import autocast
            # 清空梯度
            optimizer.zero_grad()
            # 自动混合精度前向
            with autocast():
                diffusion_loss = torch.mean(diffusion_model_train(images))
            # 缩放损失 + 反向传播
            scaler.scale(diffusion_loss).backward()
            # 更新优化器
            scaler.step(optimizer)
            # 更新缩放系数
            scaler.update()

        # -------------------#
        # 关键：每步更新 EMA 模型
        # 让生成图像更稳定、清晰
        # -------------------#
        diffusion_model.update_ema()

        # 累计损失
        total_loss += diffusion_loss.item()

        # 主进程更新进度条信息
        if local_rank == 0:
            pbar.set_postfix(
                total_loss=total_loss / (iteration + 1),
                lr=get_lr(optimizer)
            )
            pbar.update(1)

    # -------------------#
    # 单轮训练结束，计算平均损失
    # -------------------#
    avg_total_loss = total_loss / epoch_step

    # -------------------#
    # 主进程执行：日志打印 + 结果展示 + 模型保存
    # -------------------#
    if local_rank == 0:
        pbar.close()
        print(f'Epoch: {epoch + 1}/{Epoch}')
        print(f'Total_loss: {avg_total_loss:.4f}')

        # 记录损失到历史
        loss_history.append_loss(epoch + 1, total_loss=avg_total_loss)

        # 每10轮生成一次测试图像，查看生成效果
        if epoch % 10 == 0:
            print('Show_result:')
            show_result(epoch + 1, diffusion_model, images.device)

        # -------------------#
        # 模型保存策略
        # -------------------#
        # 达到保存周期 或 最后一轮 → 保存完整权重
        if (epoch + 1) % save_period == 0 or (epoch + 1) == Epoch:
            torch.save(
                diffusion_model.state_dict(),
                os.path.join(save_dir, f'Diffusion_Epoch{epoch + 1}-GLoss{avg_total_loss:.4f}.pth')
            )

        # 始终保存最新一轮权重（方便断点续训）
        torch.save(
            diffusion_model.state_dict(),
            os.path.join(save_dir, "diffusion_model_last_epoch_weights.pth")
        )