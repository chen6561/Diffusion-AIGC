import argparse
import glob
import h5py
import numpy as np
import PIL.Image as pil_image
from utils import convert_rgb_to_y


def train(args):
    """将输入图像按照设定的patch和stride大小进行裁剪，并保存为h5文件。
        Args:
            images-dir: 输入图像路径
            --output-path: 输出h5文件的路径
            --patch-size: 裁剪图像的patch大小
            --stride：裁剪时patch移动步长
            --scale: 超分辨率放大倍数
            --eval: 是否为验证集

        Returns:
            无。
    """

    h5_file = h5py.File(args.output_path, 'w')

    lr_patches = []
    hr_patches = []

    for image_path in sorted(glob.glob('{}/*'.format(args.images_dir))):
        hr = pil_image.open(image_path).convert('RGB')           # 打开图片并转为RGB格式
        hr_width = (hr.width // args.scale) * args.scale         # 取整，将图像宽度转为2的倍数
        hr_height = (hr.height // args.scale) * args.scale       # 取整，将图像高度转为2的倍数
        hr = hr.resize((hr_width, hr_height), resample=pil_image.BICUBIC) # 使用双三次插值获得高清图
        lr = hr.resize((hr_width // args.scale, hr_height // args.scale), resample=pil_image.BICUBIC) # 使用双三次插值获取低质量图
        lr = lr.resize((lr.width * args.scale, lr.height * args.scale), resample=pil_image.BICUBIC)   # 将lr图resize为与hr图像一样的大小，便于端到端训练
        hr = np.array(hr).astype(np.float32)
        lr = np.array(lr).astype(np.float32)
        hr = convert_rgb_to_y(hr) # 转为单通道图，仅使用亮度通道Y进行训练
        lr = convert_rgb_to_y(lr) # 转为单通道图，仅使用亮度通道Y进行训练

        # 根据设置的stride和patch大小，依次按照从左到右、从上到下裁剪图像
        for i in range(0, lr.shape[0] - args.patch_size + 1, args.stride):
            for j in range(0, lr.shape[1] - args.patch_size + 1, args.stride):
                lr_patches.append(lr[i:i + args.patch_size, j:j + args.patch_size])
                hr_patches.append(hr[i:i + args.patch_size, j:j + args.patch_size])

    lr_patches = np.array(lr_patches)
    hr_patches = np.array(hr_patches)

    h5_file.create_dataset('lr', data=lr_patches) # 写入到H5文件
    h5_file.create_dataset('hr', data=hr_patches) # 写入到H5文件

    h5_file.close()


def eval(args):
    h5_file = h5py.File(args.output_path, 'w')

    lr_group = h5_file.create_group('lr')
    hr_group = h5_file.create_group('hr')

    for i, image_path in enumerate(sorted(glob.glob('{}/*'.format(args.images_dir)))):
        hr = pil_image.open(image_path).convert('RGB')
        hr_width = (hr.width // args.scale) * args.scale
        hr_height = (hr.height // args.scale) * args.scale
        hr = hr.resize((hr_width, hr_height), resample=pil_image.BICUBIC)
        lr = hr.resize((hr_width // args.scale, hr_height // args.scale), resample=pil_image.BICUBIC)
        lr = lr.resize((lr.width * args.scale, lr.height * args.scale), resample=pil_image.BICUBIC)
        hr = np.array(hr).astype(np.float32)
        lr = np.array(lr).astype(np.float32)
        hr = convert_rgb_to_y(hr)
        lr = convert_rgb_to_y(lr)

        lr_group.create_dataset(str(i), data=lr)
        hr_group.create_dataset(str(i), data=hr)

    h5_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images-dir', type=str, required=True)  # 输入图像路径
    parser.add_argument('--output-path', type=str, required=True) # 输出h5文件的路径
    parser.add_argument('--patch-size', type=int, default=256)     # 裁剪图像的patch大小
    parser.add_argument('--stride', type=int, default=256)         # 裁剪时patch移动步长
    parser.add_argument('--scale', type=int, default=2)           # 超分辨率放大倍数
    parser.add_argument('--eval', action='store_true')            # 是否为验证集
    args = parser.parse_args()

    if not args.eval:
        train(args)
    else:
        eval(args)
