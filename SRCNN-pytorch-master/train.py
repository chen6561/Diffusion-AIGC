# -*- coding: utf-8 -*-
"""
SRCNN 图像超分辨率模型训练脚本
模型：SRCNN (Super-Resolution Convolutional Neural Network)
功能：训练 + 验证 + 最优模型保存
"""

import argparse
import os
import copy

import torch
from torch import nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

# 导入自定义模块
from models import SRCNN                         # SRCNN 模型结构
from datasets import TrainDataset, EvalDataset   # 训练/验证数据集
from utils import AverageMeter, calc_psnr        # 工具函数：平均损失、PSNR计算


if __name__ == '__main__':
    # ======================== 1. 命令行参数解析 ========================
    parser = argparse.ArgumentParser(description="SRCNN 模型训练脚本")
    parser.add_argument('--train-file', type=str, required=True, help="训练集 .h5 文件路径")
    parser.add_argument('--eval-file', type=str, required=True, help="验证集 .h5 文件路径")
    parser.add_argument('--outputs-dir', type=str, required=True, help="模型权重保存目录")
    parser.add_argument('--scale', type=int, default=2, help="超分放大倍数 (默认: 2)")
    parser.add_argument('--lr', type=float, default=1e-4, help="学习率 (默认: 1e-4)")
    parser.add_argument('--batch-size', type=int, default=4, help="批次大小 (默认: 4)")
    parser.add_argument('--num-epochs', type=int, default=400, help="训练轮数 (默认: 400)")
    parser.add_argument('--num-workers', type=int, default=8, help="数据加载线程数")
    parser.add_argument('--seed', type=int, default=123, help="随机种子，保证可复现")
    args = parser.parse_args()

    # ======================== 2. 输出目录设置 ========================
    # 根据放大倍数构建子目录，例如 outputs/x2
    args.outputs_dir = os.path.join(args.outputs_dir, f'x{args.scale}')

    # 如果目录不存在则创建
    if not os.path.exists(args.outputs_dir):
        os.makedirs(args.outputs_dir)

    # ======================== 3. 训练设备与随机种子配置 ========================
    # 开启cudnn加速
    cudnn.benchmark = True
    # 自动选择GPU/CPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # 设置随机种子保证训练可复现
    torch.manual_seed(args.seed)

    # ======================== 4. 模型、损失函数、优化器定义 ========================
    # 初始化SRCNN模型并迁移到设备
    model = SRCNN().to(device)
    # 使用MSE损失函数（超分经典损失）
    criterion = nn.MSELoss()
    # 分层学习率：conv1/conv2 使用基础lr，conv3 使用 0.1×lr
    optimizer = optim.Adam([
        {'params': model.conv1.parameters()},
        {'params': model.conv2.parameters()},
        {'params': model.conv3.parameters(), 'lr': args.lr * 0.1}
    ], lr=args.lr)

    # ======================== 5. 数据集与数据加载器 ========================
    # 训练数据集
    train_dataset = TrainDataset(args.train_file)
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,                # 训练集打乱
        num_workers=args.num_workers,
        pin_memory=True,             # 加速GPU数据传输
        drop_last=True               # 丢弃最后不足一个batch的数据
    )

    # 验证数据集（batch_size固定为1）
    eval_dataset = EvalDataset(args.eval_file)
    eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=1)

    # ======================== 6. 初始化最优模型参数 ========================
    best_weights = copy.deepcopy(model.state_dict())   # 最优模型权重
    best_epoch = 0                                     # 最优epoch
    best_psnr = 0.0                                    # 最高PSNR

    # ======================== 7. 训练主循环 ========================
    for epoch in range(args.num_epochs):
        # 切换为训练模式
        model.train()
        # 记录每个epoch的平均损失
        epoch_losses = AverageMeter()

        # 使用tqdm显示训练进度条
        with tqdm(total=(len(train_dataset) - len(train_dataset) % args.batch_size)) as t:
            t.set_description(f'epoch: {epoch}/{args.num_epochs - 1}')

            # 遍历训练数据
            for data in train_dataloader:
                # 获取输入（低分辨率）和标签（高分辨率）
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)

                # 前向传播：模型预测
                preds = model(inputs)

                # 计算损失
                loss = criterion(preds, labels)

                # 更新损失计算器
                epoch_losses.update(loss.item(), len(inputs))

                # 反向传播 + 参数更新
                optimizer.zero_grad()   # 清空梯度
                loss.backward()         # 反向传播
                optimizer.step()        # 更新参数

                # 更新进度条信息
                t.set_postfix(loss=f'{epoch_losses.avg:.6f}')
                t.update(len(inputs))

        # ======================== 保存当前epoch模型 ========================
        torch.save(model.state_dict(), os.path.join(args.outputs_dir, f'epoch_{epoch}.pth'))

        # ======================== 验证阶段 ========================
        model.eval()  # 切换为评估模式
        epoch_psnr = AverageMeter()

        # 验证时不计算梯度
        for data in eval_dataloader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                preds = model(inputs).clamp(0.0, 1.0)  # 输出限制在[0,1]

            # 计算并累计PSNR
            epoch_psnr.update(calc_psnr(preds, labels), len(inputs))

        # 打印当前epoch验证PSNR
        print(f'eval psnr: {epoch_psnr.avg:.2f}')

        # ======================== 更新最优模型 ========================
        if epoch_psnr.avg > best_psnr:
            best_epoch = epoch
            best_psnr = epoch_psnr.avg
            best_weights = copy.deepcopy(model.state_dict())

    # ======================== 训练结束，保存最优模型 ========================
    print(f'best epoch: {best_epoch}, psnr: {best_psnr:.2f}')
    torch.save(best_weights, os.path.join(args.outputs_dir, 'best.pth'))