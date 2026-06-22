import argparse

import torch
from torch.utils.data import DataLoader

from train import build_dataset, build_model
from utils.misc import load_config
from utils.metrics import calc_psnr, calc_ssim
from utils.image import tensor_to_img


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SwinIR")
    parser.add_argument("--config", type=str, default="configs/train_x2.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(cfg).to(device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    model.eval()

    dataset = build_dataset(cfg, is_train=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

    total_psnr = 0.0
    total_ssim = 0.0
    with torch.no_grad():
        for idx, batch in enumerate(loader):
            lr = batch["lr"].to(device)
            hr = batch["hr"]
            sr = model(lr).clamp(0, 1).cpu()

            sr_img = tensor_to_img(sr[0], rgb=hr.shape[1] == 3)
            hr_img = tensor_to_img(hr[0], rgb=hr.shape[1] == 3)
            psnr = calc_psnr(sr_img, hr_img, cfg["eval"]["crop_border"])
            ssim = calc_ssim(sr_img, hr_img, cfg["eval"]["crop_border"])
            total_psnr += psnr
            total_ssim += ssim
            print(f"{idx + 1:04d} {batch['name'][0]} PSNR={psnr:.4f} SSIM={ssim:.4f}")

    count = len(loader)
    print(f"Average PSNR={total_psnr / count:.4f}, Average SSIM={total_ssim / count:.4f}")


if __name__ == "__main__":
    main()
