import argparse
import os
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.div2k import DIV2KDataset
from datasets.medical_sr import MedicalSRDataset
from models.swinir import SwinIR
from utils.image import tensor_to_img
from utils.metrics import calc_psnr, calc_ssim
from utils.misc import AverageMeter, ensure_dir, load_config, save_checkpoint, set_random_seed


def build_dataset(cfg: Dict, is_train: bool):
    dataset_cfg = cfg["dataset"]
    dataset_type = dataset_cfg["type"].lower()
    kwargs = {
        "scale": cfg["scale"],
        "patch_size": dataset_cfg["patch_size"],
        "is_train": is_train,
        "gray": dataset_cfg.get("gray", False),
    }
    if is_train:
        hr_dir = dataset_cfg["train_hr_dir"]
        lr_dir = dataset_cfg["train_lr_dir"]
    else:
        hr_dir = dataset_cfg["val_hr_dir"]
        lr_dir = dataset_cfg["val_lr_dir"]

    if dataset_type == "medical_sr":
        return MedicalSRDataset(hr_dir=hr_dir, lr_dir=lr_dir, **kwargs)
    return DIV2KDataset(hr_dir=hr_dir, lr_dir=lr_dir, **kwargs)


def build_model(cfg: Dict) -> SwinIR:
    model_cfg = cfg["model"]
    dataset_cfg = cfg["dataset"]
    return SwinIR(
        img_size=dataset_cfg["patch_size"],
        in_chans=model_cfg["in_chans"],
        embed_dim=model_cfg["embed_dim"],
        depths=model_cfg["depths"],
        num_heads=model_cfg["num_heads"],
        window_size=model_cfg["window_size"],
        mlp_ratio=model_cfg["mlp_ratio"],
        upscale=cfg["scale"],
        upsampler=model_cfg.get("upsampler", "pixelshuffle"),
        resi_connection=model_cfg.get("resi_connection", "1conv"),
    )


def validate(model: nn.Module, loader: DataLoader, device: torch.device, crop_border: int) -> Dict[str, float]:
    model.eval()
    psnr_meter = AverageMeter()
    ssim_meter = AverageMeter()
    with torch.no_grad():
        for batch in loader:
            lr = batch["lr"].to(device)
            hr = batch["hr"].to(device)
            sr = model(lr).clamp(0, 1)

            for i in range(sr.size(0)):
                sr_img = tensor_to_img(sr[i], rgb=hr.shape[1] == 3)
                hr_img = tensor_to_img(hr[i], rgb=hr.shape[1] == 3)
                psnr_meter.update(calc_psnr(sr_img, hr_img, crop_border))
                ssim_meter.update(calc_ssim(sr_img, hr_img, crop_border))
    return {"psnr": psnr_meter.avg, "ssim": ssim_meter.avg}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SwinIR")
    parser.add_argument("--config", type=str, default="configs/train_x2.yaml")
    parser.add_argument("--resume", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_random_seed(cfg.get("seed", 42))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_set = build_dataset(cfg, is_train=True)
    val_set = build_dataset(cfg, is_train=False)
    train_loader = DataLoader(
        train_set,
        batch_size=cfg["dataset"]["batch_size"],
        shuffle=True,
        num_workers=cfg["dataset"]["num_workers"],
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        num_workers=max(1, cfg["dataset"]["num_workers"] // 2),
        pin_memory=torch.cuda.is_available(),
    )

    model = build_model(cfg).to(device)
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"].get("weight_decay", 0.0),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["train"]["epochs"])

    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = checkpoint["epoch"] + 1

    checkpoint_dir = cfg["train"]["checkpoint_dir"]
    log_dir = cfg["train"]["log_dir"]
    ensure_dir(checkpoint_dir)
    ensure_dir(log_dir)

    best_psnr = 0.0
    for epoch in range(start_epoch, cfg["train"]["epochs"]):
        model.train()
        loss_meter = AverageMeter()
        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{cfg['train']['epochs']}", leave=False)
        for batch in progress:
            lr = batch["lr"].to(device)
            hr = batch["hr"].to(device)

            sr = model(lr)
            loss = criterion(sr, hr)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            grad_clip = cfg["train"].get("grad_clip", 0.0)
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()
            loss_meter.update(loss.item(), lr.size(0))
            progress.set_postfix(loss=f"{loss_meter.avg:.4f}")

        scheduler.step()
        metrics = validate(model, val_loader, device, cfg["eval"]["crop_border"])
        current_psnr = metrics["psnr"]
        print(
            f"Epoch {epoch + 1}: "
            f"train_loss={loss_meter.avg:.6f}, "
            f"val_psnr={metrics['psnr']:.4f}, "
            f"val_ssim={metrics['ssim']:.4f}"
        )

        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "config": cfg,
        }
        latest_path = os.path.join(checkpoint_dir, "latest.pth")
        save_checkpoint(state, latest_path)

        if (epoch + 1) % cfg["train"]["save_freq"] == 0:
            save_checkpoint(state, os.path.join(checkpoint_dir, f"epoch_{epoch + 1}.pth"))

        if current_psnr > best_psnr:
            best_psnr = current_psnr
            save_checkpoint(state, os.path.join(checkpoint_dir, "best.pth"))


if __name__ == "__main__":
    main()
