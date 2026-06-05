"""
从超分辨率扩散模型中生成大批量高清图像
输入：低分辨率图像（来自 image_sample.py 生成的结果或真实低分辨率图）
输出：超分后的高分辨率图像（保存为 .npz 格式）
本脚本是 guided-diffusion 官方超分模型专用采样/推理脚本
"""

import argparse
import os

import blobfile as bf
import numpy as np
import torch as th
import torch.distributed as dist

# 分布式工具 + 日志工具
from guided_diffusion import dist_util, logger
# 超分模型工具：默认参数 + 创建模型 + 解析参数
from guided_diffusion.script_util import (
    sr_model_and_diffusion_defaults,  # 超分模型默认参数
    sr_create_model_and_diffusion,    # 创建超分模型与扩散流程
    args_to_dict,                     # 参数转字典
    add_dict_to_argparser,            # 字典参数加入解析器
)


def main():
    # ==================== 1. 解析命令行参数 ====================
    args = create_argparser().parse_args()

    # ==================== 2. 初始化分布式环境 ====================
    dist_util.setup_dist()
    # 初始化日志输出
    logger.configure()

    # ==================== 3. 创建超分模型并加载权重 ====================
    logger.log("creating model...")
    # 创建超分模型（SuperRes UNet）+ 扩散过程（去噪流程）
    model, diffusion = sr_create_model_and_diffusion(
        **args_to_dict(args, sr_model_and_diffusion_defaults().keys())
    )
    # 加载训练好的模型权重
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    # 把模型搬到 GPU/CPU
    model.to(dist_util.dev())
    # 如果开启半精度，把模型转为 FP16 格式（省显存、加速）
    if args.use_fp16:
        model.convert_to_fp16()
    # 设置模型为评估模式（禁用 dropout、batchnorm 训练特性）
    model.eval()

    # ==================== 4. 加载低分辨率输入图像 ====================
    logger.log("loading data...")
    # 从 .npz 文件读取低分辨率图像，按批次迭代
    data = load_data_for_worker(args.base_samples, args.batch_size, args.class_cond)

    # ==================== 5. 循环生成超分图像 ====================
    logger.log("creating samples...")
    all_images = []  # 保存所有生成的高清图像
    # 直到生成足够数量的样本（num_samples）
    while len(all_images) * args.batch_size < args.num_samples:
        # 取一个批次的低分辨率图像（作为超分条件）
        model_kwargs = next(data)
        # 把数据搬到模型所在设备
        model_kwargs = {k: v.to(dist_util.dev()) for k, v in model_kwargs.items()}

        # ==================== 核心：扩散模型反向采样（超分生成） ====================
        sample = diffusion.p_sample_loop(
            model,
            # 输出形状：(批次, 3通道, 高分辨率高, 高分辨率宽)
            (args.batch_size, 3, args.large_size, args.large_size),
            clip_denoised=args.clip_denoised,  # 去噪时是否裁剪像素值
            model_kwargs=model_kwargs,         # 低分辨率条件图像
        )

        # ==================== 6. 图像后处理：从 [-1,1] 转回 [0,255] ====================
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        # 从 (B, C, H, W) → (B, H, W, C)，方便保存为 numpy 图像
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        # ==================== 7. 多卡分布式聚合所有生成图像 ====================
        all_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(all_samples, sample)
        # 把所有卡生成的图像汇总
        for sample in all_samples:
            all_images.append(sample.cpu().numpy())

        logger.log(f"created {len(all_images) * args.batch_size} samples")

    # ==================== 8. 拼接并保存最终图像 ====================
    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]  # 截取需要的数量

    # 只有主进程（rank=0）执行保存
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
        logger.log(f"saving to {out_path}")
        np.savez(out_path, arr)  # 保存成 .npz 压缩文件

    # 等待所有进程同步
    dist.barrier()
    logger.log("sampling complete")


def load_data_for_worker(base_samples, batch_size, class_cond):
    """
    从 .npz 文件读取低分辨率图像，按批次迭代输出
    :param base_samples: .npz 文件路径（存放低分辨率图）
    :param batch_size: 批次大小
    :param class_cond: 是否使用类别条件
    """
    # 读取 npz 文件（支持本地/云端路径）
    with bf.BlobFile(base_samples, "rb") as f:
        obj = np.load(f)
        image_arr = obj["arr_0"]  # 低分辨率图像数组
        if class_cond:
            label_arr = obj["arr_1"]  # 类别标签（可选）

    # 分布式：当前进程编号
    rank = dist.get_rank()
    # 总进程数（GPU 数量）
    num_ranks = dist.get_world_size()

    buffer = []
    label_buffer = []

    # 无限循环迭代数据
    while True:
        # 按进程号分片读取数据（分布式数据并行）
        for i in range(rank, len(image_arr), num_ranks):
            buffer.append(image_arr[i])
            if class_cond:
                label_buffer.append(label_arr[i])

            # 攒够一个批次就输出
            if len(buffer) == batch_size:
                # 转 torch tensor
                batch = th.from_numpy(np.stack(buffer)).float()
                # 像素归一化到 [-1, 1]（扩散模型标准输入）
                batch = batch / 127.5 - 1.0
                # 形状调整：(B, H, W, C) → (B, C, H, W)
                batch = batch.permute(0, 3, 1, 2)

                # 构造模型输入：low_res 是低分辨率条件图
                res = dict(low_res=batch)
                if class_cond:
                    res["y"] = th.from_numpy(np.stack(label_buffer))

                # 迭代返回批次数据
                yield res
                buffer, label_buffer = [], []


def create_argparser():
    """
    命令行参数配置
    """
    defaults = dict(
        clip_denoised=True,        # 去噪时裁剪像素值，保证图像正常
        num_samples=10000,         # 要生成多少张图
        batch_size=16,             # 批次大小（根据显存调整）
        use_ddim=False,            # 是否使用 DDIM 快速采样
        base_samples="",           # 低分辨率图像 .npz 路径（必须输入）
        model_path="",             # 训练好的超分模型路径
    )
    # 合并超分模型默认参数（通道数、分辨率、扩散步数等）
    defaults.update(sr_model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


# 脚本入口
if __name__ == "__main__":
    main()