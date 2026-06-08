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
import os,sys
import numpy as np
from PIL import Image

# 获取当前脚本的上级目录（guided-diffusion 根目录）
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

# 导入 guided_diffusion 官方核心工具
from guided_diffusion import dist_util, logger
from guided_diffusion.image_datasets import load_data                  # 官方图像加载工具
from guided_diffusion.resample import create_named_schedule_sampler    # 扩散时间步采样器
from guided_diffusion.script_util import (
    sr_model_and_diffusion_defaults,       # 超分模型+扩散过程的默认参数
    sr_create_model_and_diffusion,         # 用于创建超分模型与扩散流程
    args_to_dict,                          # 参数对象转字典
    add_dict_to_argparser,                 # 字典参数注册到命令行解析器
)
# 导入官方训练循环核心类（封装了训练、验证、保存、日志）
from guided_diffusion.train_util import TrainLoop


# ------------------------------
# 主训练函数：整个训练流程入口
# ------------------------------
def main():
    # 1. 解析命令行参数（使用自定义的默认参数）
    args = create_argparser().parse_args()

    # 2. 初始化分布式/单卡训练环境（单卡自动适配，无需修改）
    dist_util.setup_dist()
    # 初始化日志输出配置
    logger.configure()

    # ------------------------------
    # 3. 创建超分模型 + 扩散过程
    # ------------------------------
    logger.log("creating model...")
    # 根据参数创建超分模型和扩散器（官方API）
    model, diffusion = sr_create_model_and_diffusion(
        **args_to_dict(args, sr_model_and_diffusion_defaults().keys())
    )
    # 将模型移动到指定设备（GPU/CPU）
    model.to(dist_util.dev())

    # 创建扩散模型时间步采样器（默认 uniform 均匀采样）
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    # ------------------------------
    # 4. 创建训练集数据加载器
    # ------------------------------
    logger.log("creating data loader...")
    # 加载高分辨率图像，并自动生成低分辨率图像
    data = load_superres_data(
        args.data_dir,
        args.batch_size,
        large_size=args.large_size,
        small_size=args.small_size,
        class_cond=args.class_cond,
    )

    # ------------------------------
    # 5. 创建验证集数据加载器（可选）
    # ------------------------------
    val_data = None
    # 如果配置了验证集路径，则创建验证迭代器
    if args.val_dir:
        val_data = load_superres_data(
            args.val_dir,
            batch_size=1,            # 验证每次只加载1张用于生成效果图
            large_size=args.large_size,
            small_size=args.small_size,
            class_cond=args.class_cond,
        )

    # ------------------------------
    # 6. 初始化官方训练循环 TrainLoop
    # ------------------------------
    logger.log("training...")
    trainer = TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,                 # 训练数据
        val_data=val_data,         # 验证数据（用于保存效果图）
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
    # 7. 启动训练（官方封装好的完整流程）
    # ------------------------------
    trainer.run_loop()


# ------------------------------
# 超分辨率专用数据加载函数
# 功能：输入高分辨率HR图像 → 自动下采样得到低分辨率LR图像
# 返回：高分辨率图 + 包含低分辨率图的模型参数字典
# ------------------------------
def load_superres_data(data_dir, batch_size, large_size, small_size, class_cond=False):
    # 调用官方API加载高分辨率图像
    data = load_data(
        data_dir=data_dir,
        batch_size=batch_size,
        image_size=large_size,
        class_cond=class_cond,
    )
    # 迭代读取高分辨率批次，自动生成LR图像
    for large_batch, model_kwargs in data:
        # 核心：对高分辨率图进行区域下采样，生成低分辨率图 LR
        model_kwargs["low_res"] = F.interpolate(large_batch, small_size, mode="area")
        # 返回：高分辨率图(HR) + 包含低分辨率图(LR)的参数字典
        yield large_batch, model_kwargs


# ------------------------------
# 命令行参数解析器
# 功能：统一管理所有默认参数，无需每次运行手动输入
# ------------------------------
def create_argparser():
    # 所有默认参数统一合并，无重复、无冗余，开箱即用
    defaults = {
        # 数据路径配置（直接修改这里即可）
        "data_dir": "C:/dataset/train/HR",       # 训练集：高分辨率图像路径
        "val_dir": "C:/dataset/val/HR",          # 验证集：高分辨率图像路径

        # 训练基础参数
        "batch_size": 2,                          # 批次大小（根据显存调整）
        "lr": 1e-4,                               # 学习率
        "weight_decay": 0.0,                      # 权重衰减（正则化）
        "lr_anneal_steps": 0,                     # 学习率衰减步数
        "microbatch": -1,                         # 微批次（爆显存时使用）
        "ema_rate": "0.9999",                     # EMA 模型平滑系数
        "log_interval": 10,                       # 日志打印间隔（step）
        "save_interval": 10,                    # 模型保存间隔（step）
        "resume_checkpoint": "",                  # 恢复训练的权重路径（为空则从头训练）
        "use_fp16": False,                        # 是否使用半精度训练
        "fp16_scale_growth": 1e-3,                 # 半精度损失缩放增长

        # 扩散过程参数
        "schedule_sampler": "uniform",            # 时间步采样策略
        "diffusion_steps": 1000,                  # 扩散步数
        "noise_schedule": "linear",               # 噪声调度类型

        # 超分模型结构参数（必须与推理时保持一致）
        "small_size": 64,                         # 低分辨率图像尺寸
        "large_size": 256,                        # 高分辨率图像尺寸
        "num_channels": 64,                       # 模型基础通道数
        "num_res_blocks": 2,                      # 每个模块的残差块数量
        "attention_resolutions": "32,16,8",       # 加入注意力机制的分辨率
        "class_cond": False,                      # 是否使用类别条件（超分一般关闭）
        "use_scale_shift_norm": True,             # 是否使用 scale-shift 归一化
        "dropout": 0.0,                           # dropout 概率
        "clip_denoised": True,                    # 推理时裁剪去噪结果
        "use_ddim": False,                        # 是否使用 DDIM 采样
    }

    # 合并官方默认参数（自定义参数优先级更高）
    defaults.update(sr_model_and_diffusion_defaults())

    # 创建参数解析器并注册所有参数
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


# ------------------------------
# 脚本入口
# ------------------------------
if __name__ == "__main__":
    main()