import os
from typing import Tuple

import cv2
import numpy as np
import torch


def read_image(path: str, gray: bool = False) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")

    if path.lower().endswith(".npy"):
        img = np.load(path).astype(np.float32)
        if img.ndim == 2:
            img = img[..., None]
        if img.max() > 1.0:
            img = img / 255.0
        return img

    flag = cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR
    img = cv2.imread(path, flag)
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")
    if gray:
        img = img[..., None]
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float32) / 255.0


def img_to_tensor(img: np.ndarray) -> torch.Tensor:
    if img.ndim == 2:
        img = img[..., None]
    return torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))).float()


def tensor_to_img(tensor: torch.Tensor, rgb: bool = True) -> np.ndarray:
    if tensor.ndim == 4:
        tensor = tensor.squeeze(0)
    array = tensor.detach().clamp(0, 1).cpu().numpy().transpose(1, 2, 0)
    if array.shape[2] == 1:
        return (array[:, :, 0] * 255.0).round().astype(np.uint8)
    image = (array * 255.0).round().astype(np.uint8)
    if rgb:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image


def save_image(path: str, image: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, image)


def paired_random_crop(hr: np.ndarray, lr: np.ndarray, lr_patch_size: int, scale: int) -> Tuple[np.ndarray, np.ndarray]:
    h, w = lr.shape[:2]
    if h < lr_patch_size or w < lr_patch_size:
        raise ValueError(f"LR image is smaller than patch size: {(h, w)} vs {lr_patch_size}")
    top = np.random.randint(0, h - lr_patch_size + 1)
    left = np.random.randint(0, w - lr_patch_size + 1)
    hr_patch_size = lr_patch_size * scale
    lr_patch = lr[top : top + lr_patch_size, left : left + lr_patch_size]
    hr_patch = hr[
        top * scale : top * scale + hr_patch_size,
        left * scale : left * scale + hr_patch_size,
    ]
    return hr_patch, lr_patch


def augment_pair(
    hr: np.ndarray,
    lr: np.ndarray,
    hflip: bool = True,
    vflip: bool = False,
    rot90: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    if hflip:
        hr = np.flip(hr, axis=1).copy()
        lr = np.flip(lr, axis=1).copy()
    if vflip:
        hr = np.flip(hr, axis=0).copy()
        lr = np.flip(lr, axis=0).copy()
    if rot90:
        hr = np.transpose(hr, (1, 0, 2)).copy()
        lr = np.transpose(lr, (1, 0, 2)).copy()
    return hr, lr


def mod_crop(img: np.ndarray, scale: int) -> np.ndarray:
    h, w = img.shape[:2]
    h = h - (h % scale)
    w = w - (w % scale)
    return img[:h, :w]
