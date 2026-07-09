#!/usr/bin/env python3
"""Train a SwinIR-style multitask model for SR + line-mask prediction.

Input:
    LR image

Targets:
    1) HR super-resolution label
    2) Line mask generated from the HR label

The model uses a shared SwinIR-style backbone and two heads:
    - SR head: predicts the HR grayscale image
    - Mask head: predicts the HR line-mask logits

Example:
    python train_swinir_multitask.py ^
        --train-lr-dir "D:/datasets/.../channel_in" ^
        --train-hr-dir "D:/datasets/.../channel_label" ^
        --train-mask-dir "D:/datasets/.../channel_mask" ^
        --output-dir "./outputs/swinir_multitask_run" ^
        --scale 4 --epochs 200 --batch-size 4
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset


IMG_EXTENSIONS = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}
IGNORED_FILENAMES = {"thumbs.db", ".ds_store", "desktop.ini"}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, payload: dict) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)


def is_image_file(path: Path) -> bool:
    if path.name.lower() in IGNORED_FILENAMES:
        return False
    return path.suffix.lower() in IMG_EXTENSIONS


def load_grayscale(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("L"), dtype=np.float32)


def image_to_tensor(image: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(image / 255.0).unsqueeze(0).float()


def mask_to_tensor(mask: np.ndarray) -> torch.Tensor:
    mask = (mask > 127).astype(np.float32)
    return torch.from_numpy(mask).unsqueeze(0)


def paired_files(lr_dir: Path, hr_dir: Path, mask_dir: Path) -> list[tuple[Path, Path, Path]]:
    def collect_files(root: Path) -> list[Path]:
        return sorted([p for p in root.rglob("*") if p.is_file() and is_image_file(p)])

    def unique_map(paths: list[Path], key_fn, label: str) -> dict[str, Path]:
        result: dict[str, Path] = {}
        duplicates: dict[str, list[str]] = {}
        for p in paths:
            key = key_fn(p)
            if key in result:
                duplicates.setdefault(key, [str(result[key])]).append(str(p))
            else:
                result[key] = p
        if duplicates:
            sample = {k: v[:3] for k, v in list(duplicates.items())[:5]}
            raise ValueError(f"Duplicate {label} keys found: {sample}")
        return result

    def stem_key(path: Path, is_mask: bool = False) -> str:
        key = path.stem
        if is_mask:
            for suffix in ("_mask", "-mask", "_line", "-line", "_seg", "-seg"):
                if key.endswith(suffix):
                    key = key[: -len(suffix)]
                    break
        return key

    lr_files = collect_files(lr_dir)
    hr_files = collect_files(hr_dir)
    mask_files = collect_files(mask_dir)

    # First try exact filename matching, since many industrial datasets keep names identical.
    lr_name_map = unique_map(lr_files, lambda p: p.name, "LR filename")
    hr_name_map = unique_map(hr_files, lambda p: p.name, "HR filename")
    mask_name_map = unique_map(mask_files, lambda p: p.name, "mask filename")
    common_names = sorted(set(lr_name_map.keys()) & set(hr_name_map.keys()) & set(mask_name_map.keys()))
    if common_names:
        return [(lr_name_map[name], hr_name_map[name], mask_name_map[name]) for name in common_names]

    # Fallback to stem matching for datasets where mask names carry a suffix like "_mask".
    lr_stem_map = unique_map(lr_files, lambda p: stem_key(p), "LR stem")
    hr_stem_map = unique_map(hr_files, lambda p: stem_key(p), "HR stem")
    mask_stem_map = unique_map(mask_files, lambda p: stem_key(p, is_mask=True), "mask stem")
    common_stems = sorted(set(lr_stem_map.keys()) & set(hr_stem_map.keys()) & set(mask_stem_map.keys()))
    if common_stems:
        return [(lr_stem_map[name], hr_stem_map[name], mask_stem_map[name]) for name in common_stems]

    raise FileNotFoundError(
        "No matched files were found across LR / HR / mask directories. "
        f"LR={len(lr_files)}, HR={len(hr_files)}, MASK={len(mask_files)}. "
        f"LR sample names={list(sorted(p.name for p in lr_files))[:5]}, "
        f"HR sample names={list(sorted(p.name for p in hr_files))[:5]}, "
        f"MASK sample names={list(sorted(p.name for p in mask_files))[:5]}"
    )


class PairedSRMaskDataset(Dataset):
    def __init__(
        self,
        lr_dir: Path,
        hr_dir: Path,
        mask_dir: Path,
        scale: int,
        patch_size_lr: int,
        augment: bool = True,
    ) -> None:
        self.items = paired_files(lr_dir, hr_dir, mask_dir)
        self.scale = scale
        self.patch_size_lr = patch_size_lr
        self.patch_size_hr = patch_size_lr * scale
        self.resize_size_lr = 512
        self.resize_size_hr = self.resize_size_lr * scale
        self.augment = augment

    def __len__(self) -> int:
        return len(self.items)

    def _random_crop(
        self,
        lr: np.ndarray,
        hr: np.ndarray,
        mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.patch_size_lr <= 0:
            return lr, hr, mask

        h_lr, w_lr = lr.shape
        if h_lr < self.patch_size_lr or w_lr < self.patch_size_lr:
            raise ValueError(
                f"LR patch size {self.patch_size_lr} is larger than image size {(h_lr, w_lr)}."
            )

        top_lr = random.randint(0, h_lr - self.patch_size_lr)
        left_lr = random.randint(0, w_lr - self.patch_size_lr)
        top_hr = top_lr * self.scale
        left_hr = left_lr * self.scale

        lr_patch = lr[top_lr : top_lr + self.patch_size_lr, left_lr : left_lr + self.patch_size_lr]
        hr_patch = hr[top_hr : top_hr + self.patch_size_hr, left_hr : left_hr + self.patch_size_hr]
        mask_patch = mask[top_hr : top_hr + self.patch_size_hr, left_hr : left_hr + self.patch_size_hr]
        return lr_patch, hr_patch, mask_patch

    def _augment(
        self,
        lr: np.ndarray,
        hr: np.ndarray,
        mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if random.random() < 0.5:
            lr = np.fliplr(lr).copy()
            hr = np.fliplr(hr).copy()
            mask = np.fliplr(mask).copy()
        if random.random() < 0.5:
            lr = np.flipud(lr).copy()
            hr = np.flipud(hr).copy()
            mask = np.flipud(mask).copy()
        if random.random() < 0.5:
            lr = np.rot90(lr).copy()
            hr = np.rot90(hr).copy()
            mask = np.rot90(mask).copy()
        return lr, hr, mask

    def _center_crop(self, arr: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
        h, w = arr.shape
        if h == target_h and w == target_w:
            return arr
        top = max((h - target_h) // 2, 0)
        left = max((w - target_w) // 2, 0)
        return arr[top : top + target_h, left : left + target_w]

    def _resize_image(self, arr: np.ndarray, target_h: int, target_w: int, is_mask: bool = False) -> np.ndarray:
        if arr.shape == (target_h, target_w):
            return arr
        resample = Image.Resampling.NEAREST if is_mask else Image.Resampling.BICUBIC
        pil = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
        resized = pil.resize((target_w, target_h), resample=resample)
        return np.array(resized, dtype=np.float32)

    def _align_triplet(
        self,
        lr: np.ndarray,
        hr: np.ndarray,
        mask: np.ndarray,
        name: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        lr_h, lr_w = lr.shape
        hr_h, hr_w = hr.shape
        mask_h, mask_w = mask.shape

        target_hr_h = min(lr_h * self.scale, hr_h, mask_h)
        target_hr_w = min(lr_w * self.scale, hr_w, mask_w)
        target_hr_h = (target_hr_h // self.scale) * self.scale
        target_hr_w = (target_hr_w // self.scale) * self.scale

        if target_hr_h <= 0 or target_hr_w <= 0:
            raise ValueError(
                f"Failed to align sample {name}: LR={lr.shape}, HR={hr.shape}, MASK={mask.shape}, scale={self.scale}."
            )

        target_lr_h = target_hr_h // self.scale
        target_lr_w = target_hr_w // self.scale

        if target_lr_h <= 0 or target_lr_w <= 0:
            raise ValueError(
                f"Aligned LR size became invalid for {name}: target LR {(target_lr_h, target_lr_w)}, target HR {(target_hr_h, target_hr_w)}."
            )

        lr = self._center_crop(lr, target_lr_h, target_lr_w)
        hr = self._center_crop(hr, target_hr_h, target_hr_w)
        mask = self._center_crop(mask, target_hr_h, target_hr_w)
        return lr, hr, mask

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        lr_path, hr_path, mask_path = self.items[idx]
        lr = load_grayscale(lr_path)
        hr = load_grayscale(hr_path)
        mask = load_grayscale(mask_path)

        lr, hr, mask = self._align_triplet(lr, hr, mask, lr_path.name)
        lr = self._resize_image(lr, self.resize_size_lr, self.resize_size_lr, is_mask=False)
        hr = self._resize_image(hr, self.resize_size_hr, self.resize_size_hr, is_mask=False)
        mask = self._resize_image(mask, self.resize_size_hr, self.resize_size_hr, is_mask=True)

        if self.augment:
            lr, hr, mask = self._augment(lr, hr, mask)

        sample = {
            "lr": image_to_tensor(lr),
            "hr": image_to_tensor(hr),
            "mask": mask_to_tensor(mask),
            "name": lr_path.name,
        }
        return sample


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    b, h, w, c = x.shape
    x = x.view(b, h // window_size, window_size, w // window_size, window_size, c)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, c)
    return windows


def window_reverse(windows: torch.Tensor, window_size: int, h: int, w: int) -> torch.Tensor:
    b = int(windows.shape[0] / (h * w / window_size / window_size))
    x = windows.view(b, h // window_size, w // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)
    return x


class WindowAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        window_size: int,
        num_heads: int,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        relative_coords_h = torch.arange(-(window_size - 1), window_size)
        relative_coords_w = torch.arange(-(window_size - 1), window_size)
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )

        coords = torch.stack(torch.meshgrid(torch.arange(window_size), torch.arange(window_size), indexing="ij"))
        coords_flatten = coords.flatten(1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        b_, n, c = x.shape
        input_dtype = x.dtype
        qkv = self.qkv(x).float()
        qkv = qkv.reshape(b_, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(n, n, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous().float()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            n_w = mask.shape[0]
            attn = attn.view(b_ // n_w, n_w, self.num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0).float()
            attn = attn.view(-1, self.num_heads, n, n)

        attn = attn - attn.amax(dim=-1, keepdim=True)
        attn = F.softmax(attn, dim=-1, dtype=torch.float32)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(b_, n, c)
        x = x.to(input_dtype)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int,
        shift_size: int,
        mlp_ratio: float = 2.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(
            dim=dim,
            window_size=window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), drop=drop)

    def calculate_mask(self, x_size: tuple[int, int], device: torch.device) -> torch.Tensor:
        h, w = x_size
        img_mask = torch.zeros((1, h, w, 1), device=device)
        h_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        w_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        cnt = 0
        for h_slice in h_slices:
            for w_slice in w_slices:
                img_mask[:, h_slice, w_slice, :] = cnt
                cnt += 1
        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(
            attn_mask == 0, float(0.0)
        )
        return attn_mask

    def forward(self, x: torch.Tensor, x_size: tuple[int, int]) -> torch.Tensor:
        h, w = x_size
        b, l, c = x.shape
        shortcut = x
        x = self.norm1(x)
        x = x.view(b, h, w, c)

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = self.calculate_mask(x_size, x.device)
        else:
            shifted_x = x
            attn_mask = None

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, c)
        attn_windows = self.attn(x_windows, mask=attn_mask)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, c)
        shifted_x = window_reverse(attn_windows, self.window_size, h, w)

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        x = x.view(b, h * w, c)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class BasicLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: int,
        mlp_ratio: float = 2.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: Iterable[float] | float = 0.0,
    ) -> None:
        super().__init__()
        if isinstance(drop_path, float):
            drop_path = [drop_path] * depth

        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i],
                )
                for i in range(depth)
            ]
        )

    def forward(self, x: torch.Tensor, x_size: tuple[int, int]) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x, x_size)
        return x


class RSTB(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: int,
        mlp_ratio: float = 2.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: Iterable[float] | float = 0.0,
    ) -> None:
        super().__init__()
        self.residual_group = BasicLayer(
            dim=dim,
            depth=depth,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop=drop,
            attn_drop=attn_drop,
            drop_path=drop_path,
        )
        self.conv = nn.Conv2d(dim, dim, 3, 1, 1)

    def forward(self, x: torch.Tensor, x_size: tuple[int, int]) -> torch.Tensor:
        b, l, c = x.shape
        shortcut = x
        x = self.residual_group(x, x_size)
        x_img = x.transpose(1, 2).view(b, c, x_size[0], x_size[1])
        x_img = self.conv(x_img)
        x = x_img.flatten(2).transpose(1, 2)
        return x + shortcut


class UpsampleBlock(nn.Module):
    def __init__(self, scale: int, num_feat: int) -> None:
        super().__init__()
        modules = []
        if scale == 1:
            modules += [nn.Identity()]
        elif scale in (2, 4, 8):
            for _ in range(int(math.log2(scale))):
                modules += [nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1), nn.PixelShuffle(2), nn.LeakyReLU(0.1, True)]
        elif scale == 3:
            modules += [nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1), nn.PixelShuffle(3), nn.LeakyReLU(0.1, True)]
        else:
            raise ValueError(f"Unsupported scale: {scale}")
        self.body = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class SwinIRMultiTask(nn.Module):
    """A practical SwinIR-M style backbone with SR and mask heads."""

    def __init__(
        self,
        in_chans: int = 1,
        out_chans: int = 1,
        mask_chans: int = 1,
        embed_dim: int = 180,
        depths: tuple[int, ...] = (6, 6, 6, 6, 6, 6),
        num_heads: tuple[int, ...] = (6, 6, 6, 6, 6, 6),
        window_size: int = 8,
        mlp_ratio: float = 2.0,
        scale: int = 4,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
    ) -> None:
        super().__init__()
        self.scale = scale
        self.window_size = window_size
        self.embed_dim = embed_dim

        self.conv_first = nn.Conv2d(in_chans, embed_dim, 3, 1, 1)
        dpr = torch.linspace(0, drop_path_rate, sum(depths)).tolist()

        self.layers = nn.ModuleList()
        offset = 0
        for i, depth in enumerate(depths):
            layer = RSTB(
                dim=embed_dim,
                depth=depth,
                num_heads=num_heads[i],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[offset : offset + depth],
            )
            self.layers.append(layer)
            offset += depth

        self.norm = nn.LayerNorm(embed_dim)
        self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)

        self.sr_pre_upsample = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 3, 1, 1),
            nn.LeakyReLU(0.1, True),
        )
        self.sr_upsample = UpsampleBlock(scale=scale, num_feat=embed_dim)
        self.sr_last = nn.Conv2d(embed_dim, out_chans, 3, 1, 1)

        self.mask_pre_upsample = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 3, 1, 1),
            nn.LeakyReLU(0.1, True),
        )
        self.mask_upsample = UpsampleBlock(scale=scale, num_feat=embed_dim)
        self.mask_last = nn.Conv2d(embed_dim, mask_chans, 3, 1, 1)

    def check_image_size(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        _, _, h, w = x.size()
        pad_h = (self.window_size - h % self.window_size) % self.window_size
        pad_w = (self.window_size - w % self.window_size) % self.window_size
        if pad_h != 0 or pad_w != 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        return x, (h, w)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)
        x_size = (h, w)
        for layer in self.layers:
            x = layer(x, x_size)
        x = self.norm(x)
        x = x.transpose(1, 2).view(b, self.embed_dim, h, w)
        return x

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x, original_size = self.check_image_size(x)
        x = self.conv_first(x)
        body = self.conv_after_body(self.forward_features(x)) + x

        sr = self.sr_pre_upsample(body)
        sr = self.sr_upsample(sr)
        sr = self.sr_last(sr)

        mask = self.mask_pre_upsample(body)
        mask = self.mask_upsample(mask)
        mask = self.mask_last(mask)

        h, w = original_size
        sr = sr[:, :, : h * self.scale, : w * self.scale]
        mask = mask[:, :, : h * self.scale, : w * self.scale]
        return sr, mask


class CharbonnierLoss(nn.Module):
    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        return torch.mean(torch.sqrt(diff * diff + self.eps * self.eps))


class DiceLoss(nn.Module):
    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logits = logits.float()
        target = target.float()
        probs = torch.sigmoid(logits)
        numerator = 2 * (probs * target).sum(dim=(1, 2, 3))
        denominator = probs.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + self.eps
        dice = numerator / denominator
        return 1 - dice.mean()


class EdgeLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32)
        sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32)
        self.register_buffer("sobel_x", sobel_x.view(1, 1, 3, 3))
        self.register_buffer("sobel_y", sobel_y.view(1, 1, 3, 3))

    def _grad(self, x: torch.Tensor) -> torch.Tensor:
        gx = F.conv2d(x, self.sobel_x, padding=1)
        gy = F.conv2d(x, self.sobel_y, padding=1)
        return torch.sqrt(gx * gx + gy * gy + 1e-6)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(self._grad(pred), self._grad(target))


def soft_erode(img: torch.Tensor) -> torch.Tensor:
    if img.dim() != 4:
        raise ValueError("soft_erode expects a 4D tensor [B, C, H, W].")
    p1 = -F.max_pool2d(-img, kernel_size=(3, 1), stride=1, padding=(1, 0))
    p2 = -F.max_pool2d(-img, kernel_size=(1, 3), stride=1, padding=(0, 1))
    return torch.min(p1, p2)


def soft_dilate(img: torch.Tensor) -> torch.Tensor:
    return F.max_pool2d(img, kernel_size=3, stride=1, padding=1)


def soft_open(img: torch.Tensor) -> torch.Tensor:
    return soft_dilate(soft_erode(img))


def soft_skeletonize(img: torch.Tensor, iters: int = 10) -> torch.Tensor:
    img = img.clamp(0.0, 1.0)
    skeleton = F.relu(img - soft_open(img))
    for _ in range(iters):
        img = soft_erode(img)
        opened = soft_open(img)
        delta = F.relu(img - opened)
        skeleton = skeleton + F.relu(delta - skeleton * delta)
    return skeleton


class SoftCLDiceLoss(nn.Module):
    def __init__(self, iterations: int = 10, eps: float = 1e-6) -> None:
        super().__init__()
        self.iterations = iterations
        self.eps = eps

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logits = logits.float()
        target = target.float()
        pred = torch.sigmoid(logits)
        pred_skel = soft_skeletonize(pred, self.iterations)
        target_skel = soft_skeletonize(target, self.iterations)

        tprec = (pred_skel * target).sum(dim=(1, 2, 3)) / (pred_skel.sum(dim=(1, 2, 3)) + self.eps)
        tsens = (target_skel * pred).sum(dim=(1, 2, 3)) / (target_skel.sum(dim=(1, 2, 3)) + self.eps)
        cl_dice = 2 * tprec * tsens / (tprec + tsens + self.eps)
        return 1 - cl_dice.mean()


class CombinedMaskLoss(nn.Module):
    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5) -> None:
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.bce_weight * self.bce(logits, target) + self.dice_weight * self.dice(logits, target)


@dataclass
class TrainConfig:
    train_lr_dir: Path
    train_hr_dir: Path
    train_mask_dir: Path
    val_lr_dir: Path | None
    val_hr_dir: Path | None
    val_mask_dir: Path | None
    output_dir: Path
    scale: int
    patch_size_lr: int
    batch_size: int
    epochs: int
    lr: float
    weight_decay: float
    num_workers: int
    seed: int
    amp: bool
    sr_loss_weight: float
    mask_loss_weight: float
    edge_loss_weight: float
    topo_loss_weight: float
    preview_count: int
    model_size: str
    embed_dim: int
    depths: tuple[int, ...]
    num_heads: tuple[int, ...]
    window_size: int
    checkpoint_every: int


SWINIR_MODEL_PRESETS = {
    "small": {
        "embed_dim": 60,
        "depths": (6, 6, 6, 6),
        "num_heads": (6, 6, 6, 6),
    },
    "medium": {
        "embed_dim": 180,
        "depths": (6, 6, 6, 6, 6, 6),
        "num_heads": (6, 6, 6, 6, 6, 6),
    },
    "large": {
        "embed_dim": 240,
        "depths": (6, 6, 6, 6, 6, 6),
        "num_heads": (8, 8, 8, 8, 8, 8),
    },
}


def resolve_model_preset(
    model_size: str,
    embed_dim: int,
    depths: list[int],
    num_heads: list[int],
) -> tuple[int, tuple[int, ...], tuple[int, ...]]:
    model_size = model_size.lower()
    if model_size == "custom":
        return embed_dim, tuple(depths), tuple(num_heads)
    if model_size not in SWINIR_MODEL_PRESETS:
        raise ValueError(f"Unknown --model-size: {model_size}")
    preset = SWINIR_MODEL_PRESETS[model_size]
    return preset["embed_dim"], preset["depths"], preset["num_heads"]


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-lr-dir", type=Path, required=True)
    parser.add_argument("--train-hr-dir", type=Path, required=True)
    parser.add_argument("--train-mask-dir", type=Path, required=True)
    parser.add_argument("--val-lr-dir", type=Path)
    parser.add_argument("--val-hr-dir", type=Path)
    parser.add_argument("--val-mask-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--patch-size-lr", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--model-size", type=str, default="small", choices=["small", "medium", "large", "custom"])
    parser.add_argument("--sr-loss-weight", type=float, default=1.0)
    parser.add_argument("--mask-loss-weight", type=float, default=0.5)
    parser.add_argument("--edge-loss-weight", type=float, default=0.2)
    parser.add_argument("--topo-loss-weight", type=float, default=0.2)
    parser.add_argument("--preview-count", type=int, default=4)
    parser.add_argument("--embed-dim", type=int, default=180)
    parser.add_argument("--depths", type=int, nargs="+", default=[6, 6, 6, 6, 6, 6])
    parser.add_argument("--num-heads", type=int, nargs="+", default=[6, 6, 6, 6, 6, 6])
    parser.add_argument("--window-size", type=int, default=8)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    return parser


def parse_config() -> TrainConfig:
    args = build_argparser().parse_args()
    embed_dim, depths, num_heads = resolve_model_preset(
        args.model_size,
        args.embed_dim,
        args.depths,
        args.num_heads,
    )
    if len(depths) != len(num_heads):
        raise ValueError("--depths and --num-heads must have the same length.")
    if any(depth <= 0 for depth in depths):
        raise ValueError("All depths must be positive.")
    return TrainConfig(
        train_lr_dir=args.train_lr_dir,
        train_hr_dir=args.train_hr_dir,
        train_mask_dir=args.train_mask_dir,
        val_lr_dir=args.val_lr_dir,
        val_hr_dir=args.val_hr_dir,
        val_mask_dir=args.val_mask_dir,
        output_dir=args.output_dir,
        scale=args.scale,
        patch_size_lr=args.patch_size_lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        seed=args.seed,
        amp=args.amp,
        model_size=args.model_size,
        sr_loss_weight=args.sr_loss_weight,
        mask_loss_weight=args.mask_loss_weight,
        edge_loss_weight=args.edge_loss_weight,
        topo_loss_weight=args.topo_loss_weight,
        preview_count=args.preview_count,
        embed_dim=embed_dim,
        depths=depths,
        num_heads=num_heads,
        window_size=args.window_size,
        checkpoint_every=args.checkpoint_every,
    )


def build_loaders(cfg: TrainConfig) -> tuple[DataLoader, DataLoader | None]:
    train_ds = PairedSRMaskDataset(
        lr_dir=cfg.train_lr_dir,
        hr_dir=cfg.train_hr_dir,
        mask_dir=cfg.train_mask_dir,
        scale=cfg.scale,
        patch_size_lr=cfg.patch_size_lr,
        augment=True,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = None
    if cfg.val_lr_dir and cfg.val_hr_dir and cfg.val_mask_dir:
        val_ds = PairedSRMaskDataset(
            lr_dir=cfg.val_lr_dir,
            hr_dir=cfg.val_hr_dir,
            mask_dir=cfg.val_mask_dir,
            scale=cfg.scale,
            patch_size_lr=0,
            augment=False,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=1,
            shuffle=False,
            num_workers=max(1, cfg.num_workers // 2),
            pin_memory=True,
        )

    return train_loader, val_loader


def psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = F.mse_loss(pred, target).item()
    if mse <= 1e-12:
        return 99.0
    return 10.0 * math.log10(1.0 / mse)


def mask_iou(logits: torch.Tensor, target: torch.Tensor) -> float:
    pred = (torch.sigmoid(logits) > 0.5).float()
    inter = (pred * target).sum().item()
    union = ((pred + target) > 0).float().sum().item()
    return inter / max(union, 1.0)


def tensor_to_uint8_image(x: torch.Tensor) -> np.ndarray:
    x = x.detach().float().cpu().squeeze().clamp(0, 1).numpy()
    return np.clip(x * 255.0, 0, 255).astype(np.uint8)


def save_preview_triptych(output_path: Path, lr: torch.Tensor, sr: torch.Tensor, hr: torch.Tensor) -> None:
    ensure_dir(output_path.parent)
    lr_img = tensor_to_uint8_image(lr)
    sr_img = tensor_to_uint8_image(sr)
    hr_img = tensor_to_uint8_image(hr)

    if lr_img.shape != sr_img.shape:
        lr_img = np.array(Image.fromarray(lr_img).resize((sr_img.shape[1], sr_img.shape[0]), Image.Resampling.NEAREST))
    if hr_img.shape != sr_img.shape:
        hr_img = np.array(Image.fromarray(hr_img).resize((sr_img.shape[1], sr_img.shape[0]), Image.Resampling.NEAREST))

    canvas = np.concatenate([lr_img, sr_img, hr_img], axis=1)
    Image.fromarray(canvas).save(output_path)


def validate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    sr_loss_fn: nn.Module,
    mask_loss_fn: nn.Module,
    edge_loss_fn: nn.Module,
    topo_loss_fn: nn.Module,
    cfg: TrainConfig,
    epoch: int | None = None,
) -> dict[str, float]:
    model.eval()
    totals = {"loss": 0.0, "psnr": 0.0, "iou": 0.0}
    count = 0
    preview_limit = max(cfg.preview_count, 0)
    preview_saved_count = 0
    with torch.no_grad():
        for batch in loader:
            lr = batch["lr"].to(device, non_blocking=True)
            hr = batch["hr"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            sr_pred, mask_logits = model(lr)

            sr_loss = sr_loss_fn(sr_pred.float(), hr.float())
            mask_loss = mask_loss_fn(mask_logits.float(), mask.float())
            edge_loss = edge_loss_fn(sr_pred.float(), hr.float())
            topo_loss = topo_loss_fn(mask_logits.float(), mask.float())
            loss = (
                cfg.sr_loss_weight * sr_loss
                + cfg.mask_loss_weight * mask_loss
                + cfg.edge_loss_weight * edge_loss
                + cfg.topo_loss_weight * topo_loss
            )

            totals["loss"] += loss.item()
            totals["psnr"] += psnr(sr_pred.clamp(0, 1), hr)
            totals["iou"] += mask_iou(mask_logits, mask)
            count += 1

            if preview_saved_count < preview_limit:
                batch_names = batch.get("name", [])
                max_items = min(lr.shape[0], preview_limit - preview_saved_count)
                for i in range(max_items):
                    sample_name = batch_names[i] if i < len(batch_names) else f"sample_{count:04d}_{i:02d}"
                    sample_stem = Path(sample_name).stem
                    preview_name = f"epoch_{(epoch or 0):04d}_{preview_saved_count + 1:02d}_{sample_stem}.png"
                    preview_path = cfg.output_dir / "val_previews" / preview_name
                    save_preview_triptych(preview_path, lr[i], sr_pred[i].clamp(0, 1), hr[i])
                    preview_saved_count += 1
                    if preview_saved_count >= preview_limit:
                        break

    return {k: v / max(count, 1) for k, v in totals.items()}


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    epoch: int,
    cfg: TrainConfig,
    best_val_loss: float | None,
) -> None:
    ensure_dir(path.parent)
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "config": asdict(cfg),
            "best_val_loss": best_val_loss,
        },
        path,
    )


def train() -> None:
    cfg = parse_config()
    ensure_dir(cfg.output_dir)
    save_json(cfg.output_dir / "config.json", asdict(cfg))
    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(
        f"Model preset: {cfg.model_size} | embed_dim={cfg.embed_dim} | depths={cfg.depths} | num_heads={cfg.num_heads}"
    )

    train_loader, val_loader = build_loaders(cfg)
    model = SwinIRMultiTask(
        in_chans=1,
        out_chans=1,
        mask_chans=1,
        embed_dim=cfg.embed_dim,
        depths=cfg.depths,
        num_heads=cfg.num_heads,
        window_size=cfg.window_size,
        scale=cfg.scale,
    ).to(device)

    sr_loss_fn = CharbonnierLoss().to(device)
    mask_loss_fn = CombinedMaskLoss().to(device)
    edge_loss_fn = EdgeLoss().to(device)
    topo_loss_fn = SoftCLDiceLoss(iterations=10).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=1e-6)
    scaler = GradScaler("cuda", enabled=cfg.amp and device.type == "cuda")

    best_val_loss = None
    history: list[dict[str, float | int]] = []

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_sr_loss = 0.0
        epoch_mask_loss = 0.0
        epoch_edge_loss = 0.0
        epoch_topo_loss = 0.0
        valid_batches = 0
        skipped_batches = 0
        train_preview_saved_count = 0
        start_t = time.time()

        for batch in train_loader:
            lr = batch["lr"].to(device, non_blocking=True)
            hr = batch["hr"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=cfg.amp and device.type == "cuda"): 
                sr_pred, mask_logits = model(lr)

            if not torch.isfinite(sr_pred).all() or not torch.isfinite(mask_logits).all():
                skipped_batches += 1
                bad_names = batch.get("name", [])
                print(f"Skipping non-finite model output: names={bad_names}")
                optimizer.zero_grad(set_to_none=True)
                continue

            sr_pred_loss = sr_pred.float()
            mask_logits_loss = mask_logits.float()
            hr_loss = hr.float()
            mask_loss_target = mask.float()

            sr_loss = sr_loss_fn(sr_pred_loss, hr_loss)
            mask_loss = mask_loss_fn(mask_logits_loss, mask_loss_target)
            edge_loss = edge_loss_fn(sr_pred_loss, hr_loss)
            topo_loss = topo_loss_fn(mask_logits_loss, mask_loss_target)
            loss = (
                cfg.sr_loss_weight * sr_loss
                + cfg.mask_loss_weight * mask_loss
                + cfg.edge_loss_weight * edge_loss
                + cfg.topo_loss_weight * topo_loss
            )

            if not torch.isfinite(loss):
                skipped_batches += 1
                bad_names = batch.get("name", [])
                print(
                    f"Skipping non-finite batch: names={bad_names} sr={sr_loss.item()} mask={mask_loss.item()} edge={edge_loss.item()} topo={topo_loss.item()} total={loss.item()}"
                )
                optimizer.zero_grad(set_to_none=True)
                continue

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if val_loader is None and train_preview_saved_count < max(cfg.preview_count, 0):
                batch_names = batch.get("name", [])
                max_items = min(lr.shape[0], max(cfg.preview_count, 0) - train_preview_saved_count)
                for i in range(max_items):
                    sample_name = batch_names[i] if i < len(batch_names) else f"train_sample_{epoch:04d}_{valid_batches:04d}_{i:02d}"
                    sample_stem = Path(sample_name).stem
                    preview_name = f"epoch_{epoch:04d}_{train_preview_saved_count + 1:02d}_{sample_stem}.png"
                    preview_path = cfg.output_dir / "train_previews" / preview_name
                    save_preview_triptych(preview_path, lr[i], sr_pred[i].clamp(0, 1), hr[i])
                    train_preview_saved_count += 1
                    if train_preview_saved_count >= max(cfg.preview_count, 0):
                        break

            valid_batches += 1
            epoch_loss += loss.item()
            epoch_sr_loss += sr_loss.item()
            epoch_mask_loss += mask_loss.item()
            epoch_edge_loss += edge_loss.item()
            epoch_topo_loss += topo_loss.item()

        scheduler.step()
        num_batches = max(valid_batches, 1)
        train_metrics = {
            "epoch": epoch,
            "train_loss": epoch_loss / num_batches,
            "train_sr_loss": epoch_sr_loss / num_batches,
            "train_mask_loss": epoch_mask_loss / num_batches,
            "train_edge_loss": epoch_edge_loss / num_batches,
            "train_topo_loss": epoch_topo_loss / num_batches,
            "valid_batches": valid_batches,
            "skipped_batches": skipped_batches,
            "lr": optimizer.param_groups[0]["lr"],
            "seconds": time.time() - start_t,
        }

        if val_loader is not None:
            val_metrics = validate(
                model=model,
                loader=val_loader,
                device=device,
                sr_loss_fn=sr_loss_fn,
                mask_loss_fn=mask_loss_fn,
                edge_loss_fn=edge_loss_fn,
                topo_loss_fn=topo_loss_fn,
                cfg=cfg,
                epoch=epoch,
            )
            train_metrics.update(
                {
                    "val_loss": val_metrics["loss"],
                    "val_psnr": val_metrics["psnr"],
                    "val_iou": val_metrics["iou"],
                }
            )
            is_best = best_val_loss is None or val_metrics["loss"] < best_val_loss
            if is_best:
                best_val_loss = val_metrics["loss"]
                save_checkpoint(
                    cfg.output_dir / "best_model.pth",
                    model,
                    optimizer,
                    scaler,
                    epoch,
                    cfg,
                    best_val_loss,
                )
        else:
            is_best = False

        history.append(train_metrics)
        save_json(cfg.output_dir / "history.json", {"history": history})

        if epoch % cfg.checkpoint_every == 0 or epoch == cfg.epochs:
            save_checkpoint(
                cfg.output_dir / f"checkpoint_epoch_{epoch:04d}.pth",
                model,
                optimizer,
                scaler,
                epoch,
                cfg,
                best_val_loss,
            )

        status = (
            f"Epoch {epoch:03d}/{cfg.epochs} "
            f"train_loss={train_metrics['train_loss']:.4f} "
            f"sr={train_metrics['train_sr_loss']:.4f} "
            f"mask={train_metrics['train_mask_loss']:.4f} "
            f"edge={train_metrics['train_edge_loss']:.4f} "
            f"topo={train_metrics['train_topo_loss']:.4f}"
        )
        if "val_loss" in train_metrics:
            status += (
                f" | val_loss={train_metrics['val_loss']:.4f} "
                f"val_psnr={train_metrics['val_psnr']:.2f} "
                f"val_iou={train_metrics['val_iou']:.4f}"
            )
            if is_best:
                status += " | best"
        print(status)


if __name__ == "__main__":
    train()















