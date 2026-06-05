import os
import numpy as np
import torch
from PIL import Image
import random

def _list_image_files_recursively(data_dir):
    """
    递归遍历文件夹，收集所有图片路径
    :param data_dir: 图片根目录
    :return: 所有图片的绝对路径列表（已排序）
    """
    results = []
    # os.walk 遍历根目录下所有子文件夹
    for root, _, files in os.walk(data_dir):
        # sorted 保证文件顺序一致
        for file in sorted(files):
            # 获取文件后缀名并转为小写
            ext = file.split(".")[-1].lower()
            # 只保留常见图片格式
            if ext in ["jpg", "jpeg", "png", "gif"]:
                results.append(os.path.join(root, file))
    return results

def center_crop_arr(pil_image, image_size):
    """
    中心裁剪：先等比例缩放，再从中心裁剪到目标尺寸
    :param pil_image: PIL 图像
    :param image_size: 目标尺寸（如 256）
    :return: 裁剪后的 numpy 数组 (H, W, 3)
    """
    w, h = pil_image.size
    # 计算缩放比例，保证短边等于目标尺寸
    scale = image_size / min(w, h)
    new_w, new_h = round(w * scale), round(h * scale)
    # 高保真缩放
    pil_image = pil_image.resize((new_w, new_h), Image.LANCZOS)
    # 计算中心区域坐标
    x_start = (new_w - image_size) // 2
    y_start = (new_h - image_size) // 2
    # 中心裁剪
    pil_image = pil_image.crop((x_start, y_start, x_start + image_size, y_start + image_size))
    return np.array(pil_image)

def random_crop_arr(pil_image, image_size, min_crop_frac=0.8, max_crop_frac=1.0):
    """
    随机缩放 + 随机裁剪（数据增强用）
    :param pil_image: PIL 图像
    :param image_size: 目标尺寸
    :param min_crop_frac: 最小裁剪比例
    :param max_crop_frac: 最大裁剪比例
    :return: 裁剪后的 numpy 数组 (H, W, 3)
    """
    # 随机选择一个缩放尺寸
    min_smaller_dim_size = max(image_size, int(image_size / max_crop_frac))
    max_smaller_dim_size = int(image_size / min_crop_frac)
    smaller_dim_size = random.randint(min_smaller_dim_size, max_smaller_dim_size)

    w, h = pil_image.size
    scale = smaller_dim_size / min(w, h)
    new_w, new_h = round(w * scale), round(h * scale)
    pil_image = pil_image.resize((new_w, new_h), Image.LANCZOS)

    # 随机选择裁剪起点
    x_start = random.randint(0, new_w - image_size)
    y_start = random.randint(0, new_h - image_size)
    pil_image = pil_image.crop((x_start, y_start, x_start + image_size, y_start + image_size))
    return np.array(pil_image)

class ImageDataset(torch.utils.data.Dataset):
    """
    图片数据集类，继承自 PyTorch Dataset
    负责：加载图片 → 裁剪/翻转 → 归一化 → 输出模型需要的格式
    """
    def __init__(
        self,
        resolution,        # 输出图像分辨率
        image_paths,       # 所有图片路径列表
        classes=None,      # 类别标签（分类任务用）
        shard=0,           # 多卡分布式训练用（当前不用）
        num_shards=1,      # 总 GPU 数量
        random_crop=False, # 是否开启随机裁剪
        random_flip=True,  # 是否开启随机水平翻转
    ):
        self.resolution = resolution
        # 分布式切片（单卡训练时直接取全部）
        self.local_images = image_paths[shard:][::num_shards]
        self.local_classes = classes
        self.random_crop = random_crop
        self.random_flip = random_flip

    def __len__(self):
        """数据集长度：图片数量"""
        return len(self.local_images)

    def __getitem__(self, idx):
        """
        按索引获取一条数据（核心函数）
        :param idx: 数据索引
        :return: 标准化后的图像张量 + 标签字典
        """
        # 1. 读取图片路径
        path = self.local_images[idx]
        # 2. 以 RGB 模式打开图片
        with open(path, "rb") as f:
            pil_image = Image.open(f).convert("RGB")

        # 3. 中心裁剪 / 随机裁剪
        if self.random_crop:
            arr = random_crop_arr(pil_image, self.resolution)
        else:
            arr = center_crop_arr(pil_image, self.resolution)

        # 4. 随机水平翻转（数据增强）
        if self.random_flip and random.random() < 0.5:
            arr = arr[:, ::-1]

        # 5. 归一化：从 [0, 255] → [-1, 1]（扩散模型标准格式）
        arr = arr.astype(np.float32) / 127.5 - 1.0

        # 6. 构造输出字典（支持分类任务）
        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)

        # 7. 维度转换：(H,W,3) → (3,H,W)（PyTorch 格式）
        return np.transpose(arr, [2, 0, 1]), out_dict

def load_data(
    data_dir,
    batch_size,
    image_size,
    class_cond=False,
    deterministic=False,
    random_crop=False,
    random_flip=True,
):
    """
    对外接口：创建一个无限迭代的数据加载器
    :param data_dir: 图片目录
    :param batch_size: 批次大小
    :param image_size: 图像尺寸
    :param class_cond: 是否使用类别条件
    :param deterministic: 是否固定顺序（验证/测试用）
    :param random_crop: 随机裁剪
    :param random_flip: 随机翻转
    :return: 无限数据迭代器
    """
    # 1. 递归读取所有图片路径
    all_files = _list_image_files_recursively(data_dir)
    classes = None

    # 2. 创建数据集对象
    dataset = ImageDataset(
        resolution=image_size,
        image_paths=all_files,
        classes=classes,
        shard=0,
        num_shards=1,
        random_crop=random_crop,
        random_flip=random_flip,
    )

    # 3. 创建 PyTorch DataLoader
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=not deterministic,  # 测试时关闭 shuffle
        num_workers=0,              # Windows 必须设为 0
        drop_last=True,             # 丢弃最后一个不足 batch 的数据
    )

    # 4. 无限 yield 数据（训练时需要持续取数据）
    while True:
        yield from loader