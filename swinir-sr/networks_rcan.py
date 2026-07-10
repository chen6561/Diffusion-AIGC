#!/usr/bin/env python3
from __future__ import annotations

import torch
import torch.nn as nn


def default_conv(in_channels: int, out_channels: int, kernel_size: int, bias: bool = True) -> nn.Conv2d:
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=bias)


class CALayer(nn.Module):
    def __init__(self, channel: int, reduction: int = 16) -> None:
        super().__init__()
        reduction = max(1, reduction)
        hidden = max(4, channel // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, hidden, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channel, 1, padding=0, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class RCAB(nn.Module):
    def __init__(self, num_feat: int, reduction: int = 16, res_scale: float = 1.0) -> None:
        super().__init__()
        self.body = nn.Sequential(
            default_conv(num_feat, num_feat, 3),
            nn.ReLU(inplace=True),
            default_conv(num_feat, num_feat, 3),
            CALayer(num_feat, reduction),
        )
        self.res_scale = res_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.body(x) * self.res_scale
        return res + x


class ResidualGroup(nn.Module):
    def __init__(self, num_feat: int, num_blocks: int, reduction: int = 16, res_scale: float = 1.0) -> None:
        super().__init__()
        modules = [RCAB(num_feat, reduction=reduction, res_scale=res_scale) for _ in range(num_blocks)]
        modules.append(default_conv(num_feat, num_feat, 3))
        self.body = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.body(x)
        return res + x


class UpsampleBlock(nn.Module):
    def __init__(self, scale: int, num_feat: int) -> None:
        super().__init__()
        if scale == 1:
            self.body = nn.Identity()
        elif scale in (2, 4, 8):
            modules = []
            steps = {2: 1, 4: 2, 8: 3}[scale]
            for _ in range(steps):
                modules.append(default_conv(num_feat, num_feat * 4, 3))
                modules.append(nn.PixelShuffle(2))
            self.body = nn.Sequential(*modules)
        elif scale == 3:
            self.body = nn.Sequential(default_conv(num_feat, num_feat * 9, 3), nn.PixelShuffle(3))
        else:
            raise ValueError(f"Unsupported scale for RCAN: {scale}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class RCANMultiTask(nn.Module):
    def __init__(
        self,
        in_chans: int = 1,
        out_chans: int = 1,
        mask_chans: int = 1,
        scale: int = 1,
        num_feat: int = 64,
        num_groups: int = 10,
        num_blocks: int = 20,
        reduction: int = 16,
        res_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.scale = scale
        self.head = default_conv(in_chans, num_feat, 3)
        self.body = nn.Sequential(
            *[ResidualGroup(num_feat, num_blocks, reduction=reduction, res_scale=res_scale) for _ in range(num_groups)]
        )
        self.body_tail = default_conv(num_feat, num_feat, 3)

        self.sr_pre = nn.Sequential(default_conv(num_feat, num_feat, 3), nn.ReLU(inplace=True))
        self.mask_pre = nn.Sequential(default_conv(num_feat, num_feat, 3), nn.ReLU(inplace=True))
        self.sr_up = UpsampleBlock(scale=scale, num_feat=num_feat)
        self.mask_up = UpsampleBlock(scale=scale, num_feat=num_feat)
        self.sr_last = default_conv(num_feat, out_chans, 3)
        self.mask_last = default_conv(num_feat, mask_chans, 3)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        res = self.body(x)
        res = self.body_tail(res)
        return res + x

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _, _, h, w = x.shape
        feat = self.head(x)
        body = self.forward_features(feat)

        sr = self.sr_last(self.sr_up(self.sr_pre(body)))
        mask = self.mask_last(self.mask_up(self.mask_pre(body)))

        return sr[:, :, : h * self.scale, : w * self.scale], mask[:, :, : h * self.scale, : w * self.scale]


class RCANITMultiTask(RCANMultiTask):
    def __init__(self, *args, iter_steps: int = 2, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.iter_steps = max(1, iter_steps)
        num_feat = kwargs.get("num_feat", 64)
        self.refine = nn.Sequential(default_conv(num_feat, num_feat, 3), nn.ReLU(inplace=True), default_conv(num_feat, num_feat, 3))

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        feat = x
        accum = x
        for _ in range(self.iter_steps):
            feat = self.body_tail(self.body(feat)) + x
            feat = self.refine(feat) + feat
            accum = accum + feat
        return accum / (self.iter_steps + 1)
