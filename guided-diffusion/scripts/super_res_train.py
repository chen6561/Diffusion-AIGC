"""
超分辨率扩散模型训练脚本
功能说明：
    1. 仅需输入高分辨率(HR)图像，训练中自动下采样生成低分辨率(LR)图像
    2. 模型学习：低分辨率图像 → 高分辨率图像 生成任务
    3. 支持验证集：模型每保存一次权重，自动保存一次 LR + 超分结果 + HR 拼接效果图
    4. 兼容单卡GPU训练，半精度训练
"""

# ------------------------------
# 导入依赖库
# ------------------------------
import argparse
import torch
import torch.nn.functional as F
import os
import numpy as np
from PIL import Image

# 导入guided_diffusion官方库工具
from guided_diffusion import dist_util, logger
from guided_diffusion.image_datasets import load_data
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    sr_model_and_diffusion_defaults,  # 超分模型+扩散过程默认参数
    sr_create_model_and_diffusion,    # 创建超分模型与扩散流程
    args_to_dict,                     # 命令行参数转字典
    add_dict_to_argparser,            # 字典参数注册到解析器
)
# 导入训练循环核心类
from guided_diffusion.train_util import TrainLoop


# ------------------------------
# 主训练函数
# ------------------------------
def main():
    # 1. 解析命令行参数
    args = create_argparser().parse_args()

    # 2. 初始化分布式/单卡训练环境（这里只使用单卡）
    dist_util.setup_dist()
    # 初始化日志输出
    logger.configure()

    # ------------------------------
    # 3. 创建超分模型 + 扩散过程
    # ------------------------------
    logger.log("creating model...")  # 打印日志
    # 构建模型与扩散器
    model, diffusion = sr_create_model_and_diffusion(
        **args_to_dict(args, sr_model_and_diffusion_defaults().keys())
    )
    # 将模型搬到 GPU/CPU
    model.to(dist_util.dev())

    # 创建时间步采样器（默认均匀采样）
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    # ------------------------------
    # 4. 创建训练集数据加载器
    # ------------------------------
    logger.log("creating data loader...")
    data = load_superres_data(
        args.data_dir,          # 训练集路径
        args.batch_size,        # 批次大小
        large_size=args.large_size,  # 高分辨率图像尺寸
        small_size=args.small_size,  # 低分辨率图像尺寸
        class_cond=args.class_cond,  # 是否使用类别条件（超分一般不用）
    )

    # ------------------------------
    # 5. 创建验证集数据加载器（可选）
    # ------------------------------
    val_data = None
    # 如果传入了验证集路径，则创建验证集迭代器
    if args.val_dir:
        val_data = load_superres_data(
            args.val_dir,
            batch_size=1,            # 验证每次只取1张图保存效果图
            large_size=args.large_size,
            small_size=args.small_size,
            class_cond=args.class_cond,
        )

    # ------------------------------
    # 6. 初始化训练循环 TrainLoop
    # ------------------------------
    logger.log("training...")
    trainer = TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,                 # 训练集
        val_data=val_data,         # 验证集（用于保存效果图）
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
    )

    # ------------------------------
    # 7. 启动训练
    # ------------------------------
    trainer.run_loop()


# ------------------------------
# 超分辨率专用数据加载函数
# 功能：输入HR图像 → 自动生成LR图像
# ------------------------------
def load_superres_data(data_dir, batch_size, large_size, small_size, class_cond=False):
    """
    参数说明：
        data_dir:     高分辨率图像路径
        batch_size:   批次大小
        large_size:   高分辨率尺寸
        small_size:   低分辨率尺寸
        class_cond:   是否使用类别条件（超分任务一般为False）
    """
    # 加载高分辨率图像数据集
    data = load_data(
        data_dir=data_dir,
        batch_size=batch_size,
        image_size=large_size,
        class_cond=class_cond,
    )

    # 迭代数据，自动生成低分辨率图像
    for large_batch, model_kwargs in data:
        # 核心：对高分辨率图进行下采样，生成低分辨率图 LR
        model_kwargs["low_res"] = F.interpolate(large_batch, small_size, mode="area")
        model_kwargs["low_res"] = F.interpolate(model_kwargs["low_res"], large_size, mode="bicubic")
        # 返回：高分辨率图(HR) + 包含低分辨率图(LR)的参数字典
        yield large_batch, model_kwargs


# ------------------------------
# 命令行参数解析器
# ------------------------------
def create_argparser():
    # 训练参数默认值
    defaults = dict(
        data_dir="",                    # 训练集路径（必须传入）
        val_dir="",                     # 验证集路径（可选）
        schedule_sampler="uniform",     # 时间步采样策略（默认均匀采样）
        lr=1e-4,                        # 学习率
        weight_decay=0.0,               # 权重衰减
        lr_anneal_steps=0,              # 学习率衰减步数
        batch_size=1,                   # 批次大小
        microbatch=-1,                  # 微批次（爆显存时使用）
        ema_rate="0.9999",              # EMA平滑系数
        log_interval=10,                # 日志打印间隔
        save_interval=1000,             # 模型保存间隔
        resume_checkpoint="",           # 恢复训练的权重路径
        use_fp16=False,                 # 是否开启半精度训练
        fp16_scale_growth=1e-3,         # 半精度损失缩放增长值
    )

    # 合并超分模型+扩散的默认参数
    defaults.update(sr_model_and_diffusion_defaults())

    # 创建解析器并注册参数
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


# ------------------------------
# 脚本入口
# ------------------------------
if __name__ == "__main__":
    main()