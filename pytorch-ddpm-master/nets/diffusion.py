import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial
from copy import deepcopy


def extract(a: torch.Tensor, t: torch.Tensor, x_shape: tuple) -> torch.Tensor:
    """
    从 diffusion 调度系数张量 a 中，提取对应时间步 t 的系数，并 reshape 为与输入 x 匹配的形状
    用于广播计算：[batch_size] -> [batch_size, 1, 1, 1] (适配图像4维张量)

    Args:
        a: 预计算的 diffusion 系数 (如 sqrt_alphas_cumprod, betas 等)
        t: 当前时间步张量 shape [batch_size]
        x_shape: 输入图像张量形状 [B, C, H, W]

    Returns:
        扩展形状后的系数张量，用于和图像张量做广播运算
    """
    batch_size = t.shape[0]
    # 从 a 中取出每个样本对应 t 步的系数
    out = a.gather(dim=-1, index=t)
    # 重塑为 [batch_size, 1, 1, 1]，适配图像高维广播
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


class EMA:
    """
    指数移动平均（Exponential Moving Average）
    作用：对模型权重做滑动平均，让生成图像更稳定、更清晰
    工业级/论文级生成模型必用技巧
    """

    def __init__(self, decay: float):
        self.decay = decay  # 滑动系数 一般 0.9999

    def update_average(self, old: torch.Tensor, new: torch.Tensor) -> torch.Tensor:
        """更新单组参数的EMA值"""
        if old is None:
            return new
        # EMA公式: ema_old * decay + new_param * (1 - decay)
        return old * self.decay + new * (1.0 - self.decay)

    def update_model_average(self, ema_model: nn.Module, current_model: nn.Module):
        """对整个模型的所有参数执行EMA更新"""
        for current_param, ema_param in zip(current_model.parameters(), ema_model.parameters()):
            old, new = ema_param.data, current_param.data
            ema_param.data = self.update_average(old, new)


class GaussianDiffusion(nn.Module):
    """
    DDPM 完整实现（Denoising Diffusion Probabilistic Models）
    包含两大核心流程：
        1. 前向加噪过程（固定数学公式，无需学习）
        2. 逆向去噪过程（UNet 学习预测噪声，训练核心）
    支持：条件/无条件生成、EMA 稳定生成、L1/L2 损失、余弦/线性调度
    """

    def __init__(
            self,
            model: nn.Module,  # 主干UNet模型
            img_size: int | tuple,  # 图像尺寸 (H, W)
            img_channels: int,  # 图像通道数 RGB=3
            num_classes: int = None,  # 条件生成类别数，None=无条件
            betas: list | np.ndarray = None,  # 扩散调度β序列
            loss_type: str = "l2",  # 损失函数类型 l1 / l2
            ema_decay: float = 0.9999,  # EMA衰减率
            ema_start: int = 2000,  # 多少步后开始启动EMA
            ema_update_rate: int = 1  # 每多少步更新一次EMA
    ):
        super().__init__()

        # 主模型 & EMA滑动平均模型
        self.model = model
        self.ema_model = deepcopy(model)  # EMA模型不参与梯度更新

        # EMA 配置
        self.ema = EMA(ema_decay)
        self.ema_decay = ema_decay
        self.ema_start = ema_start
        self.ema_update_rate = ema_update_rate
        self.step = 0  # 全局训练步数

        # 图像基础配置
        self.img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        self.img_channels = img_channels
        self.num_classes = num_classes

        # 损失函数校验
        if loss_type not in ["l1", "l2"]:
            raise ValueError(f"不支持的损失函数 {loss_type}, 仅支持 l1 / l2")
        self.loss_type = loss_type

        # 扩散步数 = beta序列长度
        self.num_timesteps = len(betas)

        # --------------------------
        # DDPM 核心预计算系数
        # 全部注册为buffer，随模型保存/加载
        # --------------------------
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas)  # α连乘 核心系数

        # 统一转为torch.float32
        to_torch = partial(torch.tensor, dtype=torch.float32)

        # register_buffer 用来保存【不需要梯度更新、但要随模型一起保存 / 加载的张量】
        self.register_buffer("betas", to_torch(betas))
        self.register_buffer("alphas", to_torch(alphas))
        self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))

        # √ᾱ 前向加噪系数
        self.register_buffer("sqrt_alphas_cumprod", to_torch(np.sqrt(alphas_cumprod)))
        # √(1-ᾱ) 噪声系数
        self.register_buffer("sqrt_one_minus_alphas_cumprod", to_torch(np.sqrt(1 - alphas_cumprod)))
        # 1/√α 逆向去噪系数
        self.register_buffer("reciprocal_sqrt_alphas", to_torch(np.sqrt(1.0 / alphas)))
        # 逆向去噪的噪声预测系数
        self.register_buffer("remove_noise_coeff", to_torch(betas / np.sqrt(1 - alphas_cumprod)))
        # 逆向过程的随机噪声标准差 σ=√β
        self.register_buffer("sigma", to_torch(np.sqrt(betas)))

    def update_ema(self):
        """
        每步训练后调用，更新EMA模型权重
        前期直接复制，后期启动滑动平均
        """
        self.step += 1
        if self.step % self.ema_update_rate == 0:
            if self.step < self.ema_start:
                # 初期直接赋值
                self.ema_model.load_state_dict(self.model.state_dict())
            else:
                # 启动EMA滑动平均
                self.ema.update_model_average(self.ema_model, self.model)

    # 关闭梯度计算,只用于推理 / 测试 / 生成
    @torch.no_grad()
    def remove_noise(
            self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor = None, use_ema: bool = True
    ) -> torch.Tensor:
        """
        逆向过程：单步去噪
        输入：带噪图像x_t + 时间步t
        输出：去噪一步后的图像x_{t-1}
        """
        model = self.ema_model if use_ema else self.model

        # DDPM 逆向去噪公式
        return (
                (x - extract(self.remove_noise_coeff, t, x.shape) * model(x, t, y)) *
                extract(self.reciprocal_sqrt_alphas, t, x.shape)
        )

    @torch.no_grad()
    def sample(
            self, batch_size: int, device: torch.device, y: torch.Tensor = None, use_ema: bool = True
    ) -> torch.Tensor:
        """
        从纯高斯噪声生成图像（完整T步去噪）
        Args:
            batch_size: 生成数量
            device: cuda/cpu
            y: 类别标签（条件生成）
            use_ema: 是否使用EMA模型（建议True）
        Returns:
            生成好的图像张量 [-1,1] 范围
        """
        if y is not None and batch_size != len(y):
            raise ValueError("生成数量与标签数量不匹配")

        # 初始化纯噪声 x_T ~ N(0,1)
        x = torch.randn(batch_size, self.img_channels, *self.img_size, device=device)

        # 从 T-1 → 0 逐步去噪
        for t in range(self.num_timesteps - 1, -1, -1):
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            x = self.remove_noise(x, t_batch, y, use_ema)

            # 除t=0外，每步都加随机高斯噪声（保持马尔可夫分布）
            if t > 0:
                x += extract(self.sigma, t_batch, x.shape) * torch.randn_like(x)

        return x.cpu().detach()

    @torch.no_grad()
    def sample_diffusion_sequence(
            self, batch_size: int, device: torch.device, y: torch.Tensor = None, use_ema: bool = True
    ) -> list:
        """
        生成完整去噪序列（用于可视化扩散过程）
        Returns: 每一步的图像列表
        """
        if y is not None and batch_size != len(y):
            raise ValueError("生成数量与标签数量不匹配")

        x = torch.randn(batch_size, self.img_channels, *self.img_size, device=device)
        sequence = [x.cpu().detach()]

        for t in range(self.num_timesteps - 1, -1, -1):
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            x = self.remove_noise(x, t_batch, y, use_ema)

            if t > 0:
                x += extract(self.sigma, t_batch, x.shape) * torch.randn_like(x)

            sequence.append(x.cpu().detach())

        return sequence

    def perturb_x(self, x: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """
        前向扩散核心公式：一步到位加噪到 t 步
        x_t = √ᾱ * x_0 + √(1-ᾱ) * ε
        """
        return (
                extract(self.sqrt_alphas_cumprod, t, x.shape) * x +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape) * noise
        )

    def get_losses(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor = None) -> torch.Tensor:
        """
        训练核心：
            1. 生成随机噪声
            2. 对干净图像加噪到t步
            3. 让UNet预测该噪声
            4. 计算预测噪声与真实噪声的损失
        """
        # 1. 生成真实高斯噪声
        noise = torch.randn_like(x)

        # 2. 前向加噪得到 x_t
        x_t = self.perturb_x(x, t, noise)

        # 3. UNet预测噪声
        noise_pred = self.model(x_t, t, y)

        # 4. 计算损失
        if self.loss_type == "l1":
            loss = F.l1_loss(noise_pred, noise)
        else:
            loss = F.mse_loss(noise_pred, noise)

        return loss

    def forward(self, x: torch.Tensor, y: torch.Tensor = None) -> torch.Tensor:
        """
        模型前向传播（训练入口）
        自动采样时间步 → 加噪 → 预测噪声 → 返回损失
        """
        b, c, h, w = x.shape
        if h != self.img_size[0] or w != self.img_size[1]:
            raise ValueError(f"输入尺寸({h}×{w})与模型配置({self.img_size})不匹配")

        # 随机采样时间步 t ∈ [0, T)
        t = torch.randint(0, self.num_timesteps, (b,), device=x.device, dtype=torch.long)
        return self.get_losses(x, t, y)


def generate_cosine_schedule(T: int, s: float = 0.008) -> np.ndarray:
    """
    余弦调度函数（比线性更适合图像生成，后期加噪更平滑）
    论文：Improved Denoising Diffusion Probabilistic Models
    """

    def cosine_fn(t, T):
        return (np.cos((t / T + s) / (1 + s) * np.pi / 2)) ** 2

    alphas = []
    f0 = cosine_fn(0, T)

    for t in range(T + 1):
        alphas.append(cosine_fn(t, T) / f0)

    betas = []
    for t in range(1, T + 1):
        betas.append(min(1.0 - alphas[t] / alphas[t - 1], 0.999))  # 防止β过大

    return np.array(betas)


def generate_linear_schedule(T: int, low: float = 1e-4, high: float = 2e-2) -> np.ndarray:
    """
    原始DDPM线性调度
    β从 0.0001 线性增加到 0.02
    """
    return np.linspace(low, high, T)