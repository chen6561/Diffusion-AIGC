from typing import Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modules import PatchEmbed, PatchUnEmbed, RSTB, Upsample


class SwinIR(nn.Module):
    def __init__(
        self,
        img_size: int = 64,
        in_chans: int = 3,
        embed_dim: int = 96,
        depths: Sequence[int] = (4, 4, 4, 4),
        num_heads: Sequence[int] = (6, 6, 6, 6),
        window_size: int = 8,
        mlp_ratio: float = 2.0,
        upscale: int = 2,
        upsampler: str = "pixelshuffle",
        resi_connection: str = "1conv",
    ) -> None:
        super().__init__()
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.window_size = window_size
        self.upscale = upscale
        self.upsampler = upsampler

        self.conv_first = nn.Conv2d(in_chans, embed_dim, 3, 1, 1)
        self.patch_embed = PatchEmbed()
        self.patch_unembed = PatchUnEmbed(embed_dim)

        self.layers = nn.ModuleList(
            [
                RSTB(
                    dim=embed_dim,
                    input_resolution=(img_size, img_size),
                    depth=depths[i],
                    num_heads=num_heads[i],
                    window_size=window_size,
                    mlp_ratio=mlp_ratio,
                    drop_path=0.0,
                    resi_connection=resi_connection,
                )
                for i in range(len(depths))
            ]
        )

        self.norm = nn.LayerNorm(embed_dim)
        self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)

        if upsampler != "pixelshuffle":
            raise ValueError("This educational implementation currently supports pixelshuffle only.")
        self.upsample = Upsample(upscale, embed_dim)
        self.conv_last = nn.Conv2d(embed_dim, in_chans, 3, 1, 1)

    def check_image_size(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.size()
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
        if mod_pad_h == 0 and mod_pad_w == 0:
            return x
        return F.pad(x, (0, mod_pad_w, 0, mod_pad_h), mode="reflect")

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x_size: Tuple[int, int] = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)
        for layer in self.layers:
            x = layer(x, x_size)
        x = self.norm(x)
        x = self.patch_unembed(x, x_size)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[2:]
        x = self.check_image_size(x)

        shallow = self.conv_first(x)
        deep = self.forward_features(shallow)
        body = self.conv_after_body(deep) + shallow
        out = self.upsample(body)
        out = self.conv_last(out)
        return out[:, :, : h * self.upscale, : w * self.upscale]
