import os
import random
from typing import Dict, List

import torch
from torch.utils.data import Dataset

from utils.image import augment_pair, img_to_tensor, paired_random_crop, read_image


def _collect_pairs(folder: str) -> List[str]:
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Directory not found: {folder}")
    names = []
    for name in sorted(os.listdir(folder)):
        lower = name.lower()
        if lower.endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".npy")):
            names.append(name)
    if not names:
        raise RuntimeError(f"No images found in: {folder}")
    return names


class MedicalSRDataset(Dataset):
    def __init__(
        self,
        hr_dir: str,
        lr_dir: str,
        scale: int,
        patch_size: int = 64,
        is_train: bool = True,
        gray: bool = True,
    ) -> None:
        self.hr_dir = hr_dir
        self.lr_dir = lr_dir
        self.scale = scale
        self.patch_size = patch_size
        self.is_train = is_train
        self.gray = gray
        self.names = _collect_pairs(hr_dir)

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        name = self.names[index]
        hr = read_image(os.path.join(self.hr_dir, name), gray=self.gray)
        lr = read_image(os.path.join(self.lr_dir, name), gray=self.gray)

        if self.is_train:
            hr, lr = paired_random_crop(hr, lr, self.patch_size, self.scale)
            hr, lr = augment_pair(
                hr,
                lr,
                hflip=random.random() < 0.5,
                vflip=random.random() < 0.5,
                rot90=random.random() < 0.5,
            )

        return {
            "name": name,
            "hr": img_to_tensor(hr),
            "lr": img_to_tensor(lr),
        }
