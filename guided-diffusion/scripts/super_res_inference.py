"""
从超分辨率扩散模型中生成高清图像
输入：低分辨率图像（用 OpenCV 直接读取本地图片）
输出：超分后的高分辨率图像（直接保存为 jpg/png）
本脚本是 guided-diffusion 官方超分模型专用采样/推理脚本（图片版）
"""

import argparse
import os
import glob

import blobfile as bf
import numpy as np
import torch as th
import torch.distributed as dist
import cv2

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    sr_model_and_diffusion_defaults,
    sr_create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model...")
    model, diffusion = sr_create_model_and_diffusion(
        **args_to_dict(args, sr_model_and_diffusion_defaults().keys())
    )

    # ==================== 终极修复：直接加载整个模型，无视格式 ====================
    with bf.BlobFile(args.model_path, "rb") as f:
        loaded = th.load(f, map_location="cpu")

    # 自动提取 state_dict，支持 99% 格式
    if hasattr(loaded, 'state_dict'):
        # 直接是模型对象
        checkpoint = loaded.state_dict()
    elif isinstance(loaded, list):
        # 列表包裹模型
        checkpoint = loaded[0].state_dict()
    elif isinstance(loaded, dict) and 'model' in loaded:
        # 官方格式
        checkpoint = loaded['model']
    else:
        # 直接就是 state_dict
        checkpoint = loaded

    model.load_state_dict(checkpoint)
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("loading images with OpenCV...")
    data = load_image_data(
        img_dir=args.img_dir,
        batch_size=args.batch_size,
        small_size=args.small_size
    )

    logger.log("creating super-resolution samples...")
    all_images = []
    all_names = []

    while True:
        try:
            model_kwargs, filenames = next(data)
        except StopIteration:
            break

        model_kwargs = {k: v.to(dist_util.dev()) for k, v in model_kwargs.items()}

        sample = diffusion.p_sample_loop(
            model,
            (args.batch_size, 3, args.large_size, args.large_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
        )

        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous().cpu().numpy()

        for idx in range(len(sample)):
            img_rgb = sample[idx]
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            all_images.append(img_bgr)
            all_names.append(filenames[idx])

        logger.log(f"processed {len(all_images)} images")

    if dist.get_rank() == 0:
        save_dir = args.save_dir
        os.makedirs(save_dir, exist_ok=True)

        for img, name in zip(all_images, all_names):
            save_path = os.path.join(save_dir, f"sr_{name}")
            cv2.imwrite(save_path, img)
            logger.log(f"saved: {save_path}")

        if args.save_collage:
            collage = create_collage(all_images, max_width=1600)
            collage_path = os.path.join(save_dir, "collage.jpg")
            cv2.imwrite(collage_path, collage)
            logger.log(f"collage saved: {collage_path}")

    dist.barrier()
    logger.log("sampling complete!")


def load_image_data(img_dir, batch_size, small_size):
    exts = ["jpg", "jpeg", "png", "bmp"]
    img_paths = []
    for ext in exts:
        img_paths += glob.glob(os.path.join(img_dir, f"*.{ext}"))
        img_paths += glob.glob(os.path.join(img_dir, f"*.{ext.upper()}"))

    logger.log(f"total images found: {len(img_paths)}")

    rank = dist.get_rank()
    num_ranks = dist.get_world_size()
    local_paths = img_paths[rank::num_ranks]

    buffer = []
    name_buffer = []

    for path in local_paths:
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (small_size, small_size))
        name = os.path.basename(path)

        buffer.append(img)
        name_buffer.append(name)

        if len(buffer) == batch_size:
            batch = np.stack(buffer)
            batch = th.from_numpy(batch).float()
            batch = batch / 127.5 - 1.0
            batch = batch.permute(0, 3, 1, 2)
            yield dict(low_res=batch), name_buffer
            buffer, name_buffer = [], []


def create_collage(images, max_width=1600):
    h, w = images[0].shape[:2]
    num = len(images)
    cols = max_width // w
    rows = (num + cols - 1) // cols
    collage = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
    for i, img in enumerate(images):
        y = (i // cols) * h
        x = (i % cols) * w
        collage[y:y+h, x:x+w] = img
    return collage


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        batch_size=1,
        use_ddim=False,
        img_dir="./lr_images",
        save_dir="./sr_results",
        save_collage=True,
        model_path="",
    )
    defaults.update(sr_model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()