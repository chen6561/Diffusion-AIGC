import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.transforms.functional import to_pil_image
from PIL import Image
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F

from diffusers import (
    AutoencoderKL,
    UNet2DModel,
    DDPMScheduler,
    DDIMScheduler,
)
from accelerate import Accelerator
from accelerate.utils import set_seed


# ======================
# 配置参数
# ======================
class Config:
    seed = 42
    batch_size = 4
    epochs = 500
    learning_rate = 1e-4
    # weight_decay = 1e-6
    weight_decay = 0
    gradient_accumulation_steps = 1
    mixed_precision = "fp16"

    image_size = 256
    scale_factor = 4
    latent_channels = 4
    unet_in_channels = 8
    unet_out_channels = 4

    num_train_timesteps = 1000
    num_inference_steps = 50
    beta_schedule = "linear"

    dataset_root = "C:/dataset2"
    output_dir = "./ldm_sr_results"
    save_every = 1

    vae_model_name = "stabilityai/sd-vae-ft-mse"
    pretrained_unet_path = "./ldm_sr_results/checkpoints/unet_epoch_16.pth"


config = Config()
set_seed(config.seed)


# ======================
# 感知损失（VGG，保结构）
# ======================
class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg19(pretrained=True).features[:16].eval()
        self.vgg = vgg
        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, x, y):
        return F.mse_loss(self.vgg(x), self.vgg(y))


# ======================
# 边缘损失（保直线、窗户、栏杆）
# ======================
class EdgeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.sobel_x = torch.Tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]]).view(1, 1, 3, 3)
        self.sobel_y = torch.Tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).view(1, 1, 3, 3)

    def forward(self, x, y):
        device = x.device
        sobel_x = self.sobel_x.to(device)
        sobel_y = self.sobel_y.to(device)

        def edge(img):
            img = img.mean(dim=1, keepdim=True)
            ex = F.conv2d(img, sobel_x, padding=1)
            ey = F.conv2d(img, sobel_y, padding=1)
            return torch.sqrt(ex ** 2 + ey ** 2 + 1e-8)

        return F.l1_loss(edge(x), edge(y))


# ======================
# 数据集
# ======================
class SRDataset(Dataset):
    def __init__(self, root_dir, split="train", scale_factor=4, image_size=256):
        self.root_dir = root_dir
        self.split = split
        self.scale_factor = scale_factor
        self.image_size = image_size
        self.crop_size = image_size

        self.hr_dir = os.path.join(root_dir, split, "HR")
        self.image_files = [f for f in os.listdir(self.hr_dir) if f.endswith((".png", ".jpg", ".jpeg"))]

        if split == "train":
            self.transform = transforms.Compose([
                transforms.RandomCrop(image_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize([0.5] * 3, [0.5] * 3),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5] * 3, [0.5] * 3),
            ])

        self.downsample = transforms.Resize(image_size // scale_factor, interpolation=transforms.InterpolationMode.BICUBIC)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        while True:
            img_name = self.image_files[idx]
            hr_path = os.path.join(self.hr_dir, img_name)
            hr_image = Image.open(hr_path).convert("RGB")
            w, h = hr_image.size
            if h < self.crop_size or w < self.crop_size:
                idx = np.random.randint(0, len(self.image_files))
                continue
            hr_image = self.transform(hr_image)
            lr_image = self.downsample(hr_image)
            return {"lr": lr_image, "hr": hr_image}


# ======================
# 模型构建（超分专用，无CrossAttn）
# ======================
def build_models(config):
    vae = AutoencoderKL.from_pretrained(config.vae_model_name)
    vae.requires_grad_(False)

    # 🔥 超分专用纯卷积 UNet，无交叉注意力，结构绝对保真
    unet = UNet2DModel(
        sample_size=config.image_size // 8,
        in_channels=config.unet_in_channels,
        out_channels=config.unet_out_channels,
        layers_per_block=2,
        block_out_channels=(128, 256, 512, 512),
        down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"),
    )

    noise_scheduler = DDPMScheduler(num_train_timesteps=config.num_train_timesteps, beta_schedule=config.beta_schedule, prediction_type="epsilon")
    inference_scheduler = DDIMScheduler(num_train_timesteps=config.num_train_timesteps, beta_schedule=config.beta_schedule, prediction_type="epsilon", clip_sample=False)

    return vae, unet, noise_scheduler, inference_scheduler


# ======================
# 训练（结构损失 + 感知损失 + 边缘损失）
# ======================
def train(config):
    accelerator = Accelerator(gradient_accumulation_steps=config.gradient_accumulation_steps, mixed_precision=config.mixed_precision, log_with="tensorboard", project_dir=os.path.join(config.output_dir, "logs"))
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(os.path.join(config.output_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(config.output_dir, "samples"), exist_ok=True)

    train_dataset = SRDataset(root_dir=config.dataset_root, split="train", scale_factor=config.scale_factor, image_size=config.image_size)
    val_dataset = SRDataset(root_dir=config.dataset_root, split="val", scale_factor=config.scale_factor, image_size=config.image_size)
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    vae, unet, noise_scheduler, _ = build_models(config)
    perceptual_loss = PerceptualLoss().to(accelerator.device)
    edge_loss = EdgeLoss().to(accelerator.device)

    if config.pretrained_unet_path is not None and os.path.exists(config.pretrained_unet_path):
        accelerator.print(f"✅ 加载预训练UNet: {config.pretrained_unet_path}")
        unet.load_state_dict(torch.load(config.pretrained_unet_path, map_location="cpu"))

    optimizer = optim.AdamW(unet.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    unet, optimizer, train_dataloader, val_dataloader = accelerator.prepare(unet, optimizer, train_dataloader, val_dataloader)
    vae = vae.to(accelerator.device)

    with torch.no_grad():
        sample_hr = next(iter(train_dataloader))["hr"][:1].to(accelerator.device)
        sample_latent = vae.encode(sample_hr).latent_dist.sample()
        vae_scale_factor = 1 / torch.std(sample_latent)
        accelerator.print(f"VAE缩放因子: {vae_scale_factor:.4f}")

    global_step = 0
    for epoch in range(config.epochs):
        unet.train()
        total_loss = 0.0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{config.epochs}")

        for batch in progress_bar:
            with accelerator.accumulate(unet):
                lr_images = batch["lr"]
                hr_images = batch["hr"]

                with torch.no_grad():
                    hr_latents = vae.encode(hr_images).latent_dist.sample() * vae_scale_factor
                    lr_latents = vae.encode(lr_images).latent_dist.sample() * vae_scale_factor

                # 🔥 移除可学习上采样，直接使用 bilinear 上采样
                lr_latents_up = F.interpolate(lr_latents, scale_factor=4, mode="bilinear", align_corners=False)

                noise = torch.randn_like(hr_latents)
                timesteps = torch.randint(0, config.num_train_timesteps, (hr_latents.shape[0],), device=hr_latents.device)
                noisy_latents = noise_scheduler.add_noise(hr_latents, noise, timesteps)
                unet_input = torch.cat([noisy_latents, lr_latents_up], dim=1)

                noise_pred = unet(unet_input, timesteps).sample

                # 🔥 三合一损失：保真 + 结构 + 边缘
                loss_noise = F.mse_loss(noise_pred, noise)
                pred_img = vae.decode(noise_pred / vae_scale_factor).sample
                gt_img = vae.decode(hr_latents / vae_scale_factor).sample
                loss_per = perceptual_loss(pred_img, gt_img)
                loss_edg = edge_loss(pred_img, gt_img)

                loss = loss_noise + 0.00 * loss_per + 0.00 * loss_edg

                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
            global_step += 1

        avg_loss = total_loss / len(train_dataloader)
        accelerator.print(f"Epoch {epoch + 1} 平均损失: {avg_loss:.4f}")

        if (epoch + 1) % config.save_every == 0 or epoch == config.epochs - 1:
            save_path = os.path.join(config.output_dir, "checkpoints", f"unet_epoch_{epoch + 1}.pth")
            accelerator.save(accelerator.unwrap_model(unet).state_dict(), save_path)
            accelerator.print(f"模型已保存到: {save_path}")
            generate_samples(config, accelerator, vae, unet, val_dataloader, epoch + 1, vae_scale_factor)

    accelerator.print("训练完成!")


# ======================
# 生成
# ======================
@torch.no_grad()
def generate_samples(config, accelerator, vae, unet, dataloader, epoch, vae_scale_factor):
    unet.eval()
    _, _, _, inference_scheduler = build_models(config)
    inference_scheduler.set_timesteps(config.num_inference_steps)

    batch = next(iter(dataloader))
    lr_images = batch["lr"][:4].to(accelerator.device)
    hr_images = batch["hr"][:4].to(accelerator.device)

    lr_latents = vae.encode(lr_images).latent_dist.sample() * vae_scale_factor
    # 🔥 直接 bilinear 上采样
    lr_latents_up = F.interpolate(lr_latents, scale_factor=4, mode="bilinear", align_corners=False)
    latents = torch.randn_like(lr_latents_up)

    for t in tqdm(inference_scheduler.timesteps, desc="生成样本"):
        unet_input = torch.cat([latents, lr_latents_up], dim=1)
        noise_pred = unet(unet_input, t).sample
        latents = inference_scheduler.step(noise_pred, t, latents).prev_sample

    sr_images = vae.decode(latents / vae_scale_factor).sample
    save_path = os.path.join(config.output_dir, "samples", f"epoch_{epoch}.png")
    plot_results(lr_images, sr_images, hr_images, save_path, config.scale_factor)


def plot_results(lr_images, sr_images, hr_images, save_path, scale_factor):
    lr_images = (lr_images * 0.5 + 0.5).clamp(0, 1).cpu()
    sr_images = (sr_images * 0.5 + 0.5).clamp(0, 1).cpu()
    hr_images = (hr_images * 0.5 + 0.5).clamp(0, 1).cpu()
    lr_upsampled = F.interpolate(lr_images, size=hr_images.shape[2:], mode="bilinear", align_corners=False)

    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    for i in range(4):
        axes[i, 0].imshow(to_pil_image(lr_images[i]))
        axes[i, 0].set_title("LR")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(to_pil_image(lr_upsampled[i]))
        axes[i, 1].set_title(f"Bilinear x{scale_factor}")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(to_pil_image(sr_images[i]))
        axes[i, 2].set_title(f"LDM SR x{scale_factor}")
        axes[i, 2].axis("off")

        axes[i, 3].imshow(to_pil_image(hr_images[i]))
        axes[i, 3].set_title("HR")
        axes[i, 3].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# ======================
# 单张图推理
# ======================
@torch.no_grad()
def super_resolve_single_image(image_path, unet_path, config, device="cuda"):
    vae = AutoencoderKL.from_pretrained(config.vae_model_name).to(device)
    unet = UNet2DModel(
        sample_size=config.image_size // 8, in_channels=8, out_channels=4,
        layers_per_block=2, block_out_channels=(128, 256, 512, 512),
        down_block_types=("DownBlock2D",) * 4, up_block_types=("UpBlock2D",) * 4
    ).to(device)
    unet.load_state_dict(torch.load(unet_path, map_location=device))

    scheduler = DDIMScheduler(num_train_timesteps=config.num_train_timesteps, beta_schedule=config.beta_schedule, prediction_type="epsilon", clip_sample=False)
    scheduler.set_timesteps(config.num_inference_steps)

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5] * 3, [0.5] * 3)])
    lr_image = Image.open(image_path).convert("RGB")
    lr_tensor = transform(lr_image).unsqueeze(0).to(device)

    with torch.no_grad():
        vae_scale_factor = 1 / torch.std(vae.encode(torch.randn(1, 3, 256, 256).to(device)).latent_dist.sample())

    lr_latents = vae.encode(lr_tensor).latent_dist.sample() * vae_scale_factor
    # 🔥 直接 bilinear 上采样
    lr_latents_up = F.interpolate(lr_latents, scale_factor=4, mode="bilinear", align_corners=False)
    latents = torch.randn_like(lr_latents_up)

    for t in tqdm(scheduler.timesteps, desc="超分处理中"):
        noise_pred = unet(torch.cat([latents, lr_latents_up], dim=1), t).sample
        latents = scheduler.step(noise_pred, t, latents).prev_sample

    sr_tensor = vae.decode(latents / vae_scale_factor).sample
    sr_image = to_pil_image((sr_tensor[0] * 0.5 + 0.5).clamp(0, 1))
    return sr_image


# ======================
# 主函数
# ======================
if __name__ == "__main__":
    train(config)

    # 推理示例（已简化，无需上采样模型）
    # sr_image = super_resolve_single_image(
    #     image_path="./test.png",
    #     unet_path="./ldm_sr_results/checkpoints/unet_epoch_100.pth",
    #     config=config
    # )
    # sr_image.save("sr_result.png")