import math

import numpy as np
from skimage.metrics import structural_similarity


def calc_psnr(sr: np.ndarray, hr: np.ndarray, crop_border: int = 0) -> float:
    if crop_border > 0:
        sr = sr[crop_border:-crop_border, crop_border:-crop_border, ...]
        hr = hr[crop_border:-crop_border, crop_border:-crop_border, ...]
    sr = sr.astype(np.float64)
    hr = hr.astype(np.float64)
    mse = np.mean((sr - hr) ** 2)
    if mse == 0:
        return float("inf")
    return 20.0 * math.log10(255.0 / math.sqrt(mse))


def calc_ssim(sr: np.ndarray, hr: np.ndarray, crop_border: int = 0) -> float:
    if crop_border > 0:
        sr = sr[crop_border:-crop_border, crop_border:-crop_border, ...]
        hr = hr[crop_border:-crop_border, crop_border:-crop_border, ...]

    if sr.ndim == 2:
        return float(structural_similarity(sr, hr, data_range=255))
    if sr.shape[2] == 1:
        return float(structural_similarity(sr[:, :, 0], hr[:, :, 0], data_range=255))
    return float(structural_similarity(sr, hr, channel_axis=2, data_range=255))
