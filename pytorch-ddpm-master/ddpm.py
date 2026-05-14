# 导入PyTorch核心库
import torch
# 导入神经网络模块
import torch.nn as nn
# 导入函数式API
import torch.nn.functional as F
# 导入配置类
from config import DDPMConfig
# 导入UNet模型主体
from models import UNet


class DDPM(nn.Module):
    """
    DDPM（Denoising Diffusion Probabilistic Models）核心实现类
    包含：前向扩散过程、反向采样过程、损失计算
    """

    def __init__(self, config: DDPMConfig):
        """
        初始化DDPM模型
        :param config: 配置对象，包含所有超参数
        """
        super().__init__()
        # 保存配置
        self.config = config
        # 初始化UNet模型，并移动到指定设备
        self.model = UNet(config).to(config.device)

        # ===================== 扩散过程参数预计算 =====================
        # 生成线性递增的beta序列：从beta_start到beta_end，共num_timesteps步
        self.betas = torch.linspace(
            config.beta_start, config.beta_end, config.num_timesteps, device=config.device
        )
        # alpha = 1 - beta
        self.alphas = 1.0 - self.betas
        # alpha的累积乘积（bar_alpha_t）
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        # 上一时刻的累积alpha乘积，t=0时为1.0
        self.alphas_cumprod_prev = torch.cat([
            torch.tensor([1.0], device=config.device), self.alphas_cumprod[:-1]
        ])

        # ===================== 扩散过程常用系数预计算 =====================
        # sqrt(bar_alpha_t)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        # sqrt(1 - bar_alpha_t)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        # log(1 - bar_alpha_t)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
        # sqrt(1 / bar_alpha_t)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        # sqrt(1 / bar_alpha_t - 1)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None) -> torch.Tensor:
        """
        前向扩散过程（训练用）：从清晰图像x0逐步加噪得到xt
        q(xt | x0) = N(xt; sqrt(bar_alpha_t)*x0, (1-bar_alpha_t)*I)
        """
        # 如果没有传入噪声，则生成标准正态噪声
        if noise is None:
            noise = torch.randn_like(x0)

        # 提取当前批次对应时间步的系数，并reshape为[B,1,1,1]以匹配图像维度
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x0.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape)

        # 重参数化生成xt
        return sqrt_alphas_cumprod_t * x0 + sqrt_one_minus_alphas_cumprod_t * noise

    def predict_noise(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """UNet模型预测噪声"""
        return self.model(x, t)

    def p_sample(self, xt, t):
        """
        单步反向采样（推理用）
        从 xt 预测 xt_prev（t-1时刻图像）
        """
        # 获取批次大小
        batch_size = xt.shape[0]

        # UNet预测当前时刻噪声
        noise_pred = self.model(xt, t)

        # 提取当前时间步对应的alpha、累积alpha、beta（适配batch维度）
        alpha_t = self._extract(self.alphas, t, xt.shape)
        alphas_cumprod_t = self._extract(self.alphas_cumprod, t, xt.shape)
        betas_t = self._extract(self.betas, t, xt.shape)

        # 计算反向分布的均值（核心公式）
        mean = (xt - (1 - alpha_t) / torch.sqrt(1 - alphas_cumprod_t + 1e-8) * noise_pred) / torch.sqrt(alpha_t)

        # 方差直接使用beta_t
        var = betas_t
        # t=0时刻方差强制为0，不添加噪声
        var = torch.where(t.reshape(-1, 1, 1, 1) == 0, torch.tensor(0.0, device=xt.device), var)

        # 生成随机噪声，t=0时为0
        noise = torch.randn_like(xt)
        noise = torch.where(t.reshape(-1, 1, 1, 1) == 0, torch.zeros_like(xt), noise)

        # 重参数化得到xt_prev
        return mean + torch.sqrt(var + 1e-8) * noise

    def p_sample_loop(self, batch_size: int) -> torch.Tensor:
        """
        完整反向采样循环：从纯高斯噪声生成清晰图像
        从 T → 0 逐步去噪
        """
        # 推理阶段不计算梯度
        with torch.no_grad():
            # 初始化纯高斯噪声xt
            xt = torch.randn(
                batch_size, self.config.in_channels,
                self.config.image_size, self.config.image_size,
                device=self.config.device
            )

            # 从T-1步倒序推理到0步
            for t in reversed(range(0, self.config.num_timesteps)):
                # 构造当前批次的时间步张量
                t_tensor = torch.full((batch_size,), t, device=self.config.device, dtype=torch.long)
                # 执行单步去噪
                xt = self.p_sample(xt, t_tensor)

            # 将输出限制在[-1,1]范围（符合归一化图像格式）
            return xt.clamp(-1.0, 1.0)

    def loss(self, x0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        DDPM训练损失：预测噪声与真实噪声的MSE损失
        这是DDPM最核心的简化损失函数
        """
        # 生成真实高斯噪声
        noise = torch.randn_like(x0)

        # 前向扩散：得到xt
        xt = self.q_sample(x0, t, noise)

        # 模型预测噪声
        noise_pred = self.predict_noise(xt, t)

        # 计算均方误差损失
        return F.mse_loss(noise_pred, noise)

    def _extract(self, arr: torch.Tensor, t: torch.Tensor, shape: torch.Size) -> torch.Tensor:
        """
        工具函数：从一维数组中按时间步t提取对应的值，并reshape为[B,1,1,1]
        用于匹配图像张量的维度
        """
        out = arr.gather(0, t)
        return out.reshape(shape[0], *([1] * (len(shape) - 1)))