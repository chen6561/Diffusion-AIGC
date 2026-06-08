"""
超分辨率扩散模型 单卡推理脚本
兼容 guided-diffusion 官方超分模型
支持命令行参数 / 批量图片推理
功能：输入低分辨率图片文件夹 → 输出超分辨率高清图片
"""
import argparse
import os
import glob
import cv2
import numpy as np
import torch as th

# 导入guided_diffusion官方核心工具
from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    sr_model_and_diffusion_defaults,  # 超分模型+扩散过程默认参数
    sr_create_model_and_diffusion,    # 创建超分模型与扩散流程
    args_to_dict,                     # 参数转换工具
    add_dict_to_argparser,            # 命令行参数注册
)


def create_argparser():
    """
    创建命令行参数解析器
    注册推理所需的所有配置项，与训练脚本保持一致
    """
    # 推理相关默认参数
    defaults = dict(
        clip_denoised=True,            # 推理时裁剪去噪结果，保证图像正常
        batch_size=1,                  # 推理批次大小
        use_ddim=False,                # 是否使用DDIM快速采样
        use_fp16=False,                # 是否使用半精度推理
        img_dir="./lr_images",         # 低分辨率图片输入目录
        save_dir="./sr_results",       # 高清图片输出目录
        model_path="./results/model179000.pt", # 训练好的模型权重路径
        small_size = 128,              # 低清图像尺寸
        large_size = 512               # 高清图像尺寸

    )
    # 合并超分模型官方默认参数
    defaults.update(sr_model_and_diffusion_defaults())
    # 创建解析器并注册所有参数
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


def main():
    """主函数：推理全流程"""
    # 1. 解析命令行参数
    args = create_argparser().parse_args()

    # 2. 初始化分布式环境（单卡自动适配）
    dist_util.setup_dist()
    # 初始化日志输出
    logger.configure()

    # ==================== 步骤1：创建超分模型与扩散过程 ====================
    logger.log("创建模型...")
    model, diffusion = sr_create_model_and_diffusion(
        **args_to_dict(args, sr_model_and_diffusion_defaults().keys())
    )

    # ==================== 步骤2：加载训练好的模型权重 ====================
    logger.log(f"加载模型: {args.model_path}")
    # 读取权重文件
    with open(args.model_path, "rb") as f:
        loaded = th.load(f, map_location="cpu")

    # 自动兼容多种权重存储格式（官方/自定义训练都支持）
    if hasattr(loaded, 'state_dict'):
        # 加载的是完整模型对象，提取参数
        checkpoint = loaded.state_dict()
    elif isinstance(loaded, list):
        # 权重被列表包裹
        checkpoint = loaded[0].state_dict()
    elif isinstance(loaded, dict) and 'model' in loaded:
        # 官方标准格式：字典包含model键
        checkpoint = loaded['model']
    else:
        # 直接就是参数字典
        checkpoint = loaded

    # 将权重加载到模型
    model.load_state_dict(checkpoint)
    # 模型移至计算设备（GPU/CPU）
    model.to(dist_util.dev())
    # 如果开启半精度，转换模型
    if args.use_fp16:
        model.convert_to_fp16()
    # 设置模型为推理模式（关闭dropout、bn等）
    model.eval()

    # ==================== 步骤3：扫描输入图片文件夹 ====================
    # 支持的图片格式
    exts = ["jpg", "jpeg", "png", "bmp", "JPG", "JPEG", "PNG", "BMP"]
    img_paths = []
    # 遍历所有格式，收集图片路径
    for ext in exts:
        img_paths += glob.glob(os.path.join(args.img_dir, f"*.{ext}"))

    logger.log(f"总共找到 {len(img_paths)} 张图片")

    # ==================== 步骤4：创建输出文件夹 ====================
    os.makedirs(args.save_dir, exist_ok=True)

    # ==================== 步骤5：逐张图片推理+保存 ====================
    for path in img_paths:
        # ------------------- 图片预处理（与训练时完全一致） -------------------
        # 1. 用OpenCV读取图片（默认BGR格式）
        img = cv2.imread(path)
        # 2. BGR转RGB（模型训练用RGB）
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # 3. 缩放到模型要求的低分辨率尺寸
        img = cv2.resize(img, (args.small_size, args.small_size))
        # 4. 转为张量 + 归一化到 [-1, 1] 区间（扩散模型标准格式）
        img = th.from_numpy(img).float() / 127.5 - 1.0
        # 5. 调整维度：(H,W,C) → (1, C, H, W)，增加batch维度，并移至设备
        img = img.permute(2, 0, 1).unsqueeze(0).to(dist_util.dev())

        # 构造模型输入参数
        model_kwargs = {"low_res": img}

        # ------------------- 核心：扩散模型采样推理 -------------------
        logger.log(f"推理: {os.path.basename(path)}")
        sample = diffusion.p_sample_loop(
            model,                      # 加载好的超分模型
            # 输出形状：(batch, 通道, 高分辨率尺寸, 高分辨率尺寸)
            (1, 3, args.large_size, args.large_size),
            clip_denoised=args.clip_denoised,  # 裁剪去噪结果
            model_kwargs=model_kwargs,        # 输入低分辨率图
        )

        # ------------------- 推理结果后处理 -------------------
        # 1. 从 [-1,1] 还原回 [0,255] 像素值
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        # 2. 调整维度：(1,C,H,W) → (H,W,C)，转为numpy格式
        sample = sample.permute(0, 2, 3, 1)[0].cpu().numpy()
        # 3. RGB转回BGR（OpenCV保存需要）
        sample = cv2.cvtColor(sample, cv2.COLOR_RGB2BGR)

        # ------------------- 保存高清图片 -------------------
        save_name = f"sr_{os.path.basename(path)}"
        save_path = os.path.join(args.save_dir, save_name)
        cv2.imwrite(save_path, sample)
        logger.log(f"已保存: {save_path}")

    # 全部推理完成
    logger.log("\n✅ 全部推理完成！")


if __name__ == "__main__":
    main()