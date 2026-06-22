import argparse
import os

import torch

from models.swinir import SwinIR
from utils.image import img_to_tensor, read_image, save_image, tensor_to_img
from utils.misc import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inference with SwinIR")
    parser.add_argument("--config", type=str, default="configs/train_x2.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    model_cfg = cfg["model"]
    gray = cfg["dataset"].get("gray", False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SwinIR(
        img_size=cfg["dataset"]["patch_size"],
        in_chans=model_cfg["in_chans"],
        embed_dim=model_cfg["embed_dim"],
        depths=model_cfg["depths"],
        num_heads=model_cfg["num_heads"],
        window_size=model_cfg["window_size"],
        mlp_ratio=model_cfg["mlp_ratio"],
        upscale=cfg["scale"],
        upsampler=model_cfg.get("upsampler", "pixelshuffle"),
        resi_connection=model_cfg.get("resi_connection", "1conv"),
    ).to(device)

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    model.eval()

    image = read_image(args.input, gray=gray)
    tensor = img_to_tensor(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor).clamp(0, 1)

    output_img = tensor_to_img(output[0], rgb=not gray)
    save_image(args.output, output_img)
    print(f"Saved SR image to: {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
