import os
import random
from typing import Dict, List

import torch
from torch.utils.data import Dataset

from utils.image import augment_pair, img_to_tensor, paired_random_crop, read_image


def _list_image_files(folder: str) -> List[str]:
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Directory not found: {folder}")
    names = []
    for name in sorted(os.listdir(folder)):
        lower = name.lower()
        if lower.endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
            names.append(name)
    if not names:
        raise RuntimeError(f"No images found in: {folder}")
    return names


class DIV2KDataset(Dataset):
    def __init__(
        self,
        hr_dir: str,
        lr_dir: str,
        scale: int,
        patch_size: int = 64,
        is_train: bool = True,
        gray: bool = False,
    ) -> None:
        self.hr_dir = hr_dir
        self.lr_dir = lr_dir
        self.scale = scale
        self.patch_size = patch_size
        self.is_train = is_train
        self.gray = gray
        self.names = _list_image_files(hr_dir)

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        name = self.names[index]
        hr_path = os.path.join(self.hr_dir, name)
        lr_path = os.path.join(self.lr_dir, name)
        hr = read_image(hr_path, gray=self.gray)
        lr = read_image(lr_path, gray=self.gray)

        if self.is_train:
            hr, lr = paired_random_crop(hr, lr, self.patch_size, self.scale)
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            rot90 = random.random() < 0.5
            hr, lr = augment_pair(hr, lr, hflip=hflip, vflip=vflip, rot90=rot90)

        return {
            "name": name,
            "hr": img_to_tensor(hr),
            "lr": img_to_tensor(lr),
        }
