# 导入PyTorch核心库
import torch
# 导入神经网络层模块
import torch.nn as nn
# 导入激活函数、卷积等函数式接口
import torch.nn.functional as F


# ====================== 模型配置类 ======================
class DDPMConfig:
    """
    DDPM模型基础配置类
    定义UNet结构、时间嵌入、图像尺寸等所有超参数
    """

    def __init__(self):
        # 输入图像通道数（RGB=3）
        self.in_channels = 3
        # UNet基础通道数（模型宽度）
        self.base_channels = 64
        # 下采样/上采样通道倍数
        self.channel_mults = [1, 2, 4, 8]
        # Dropout概率，防止过拟合
        self.dropout = 0.1
        # 输入图像尺寸（正方形）
        self.image_size = 64
        # 时间嵌入（Time Embedding）向量维度
        self.time_emb_dim = 256


# ====================== 时间嵌入模块 ======================
class TimeEmbedding(nn.Module):
    """
    时间步嵌入层
    将离散的时间步 t 转换为连续的高维向量，供UNet使用
    使用正弦/余弦位置编码 + MLP非线性变换
    """

    def __init__(self, dim):
        super().__init__()
        # 保存时间嵌入维度
        self.dim = dim
        # 定义MLP层：对时间编码做非线性变换
        self.mlp = nn.Sequential(
            nn.Linear(dim // 2, dim),  # 输入：sin+cos拼接向量 → 升维到目标dim
            nn.SiLU(),  # 激活函数
            nn.Linear(dim, dim)  # 再次线性变换
        )

    def forward(self, t):
        """
        前向传播：将时间步 t 转换为嵌入向量
        t: [batch_size] 时间步张量
        返回：[batch_size, time_emb_dim] 时间嵌入向量
        """
        # 计算频率维度（取嵌入维度的1/4）
        half = self.dim // 4
        # 生成频率系数
        emb = torch.exp(torch.arange(half, device=t.device) *
                        (-torch.log(torch.tensor(10000.0)) / (half - 1)))
        # 外积：[batch, 1] * [1, half] → [batch, half]
        emb = t[:, None] * emb[None, :]
        # 拼接sin和cos编码，得到 [batch, half*2]
        emb = torch.cat([emb.sin(), emb.cos()], 1)
        # 经过MLP输出最终时间嵌入
        return self.mlp(emb)


# ====================== 残差块 ======================
class ResBlock(nn.Module):
    """
    残差连接块（Residual Block）
    包含卷积、时间嵌入融合、残差shortcut
    """

    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        # 第一个3x3卷积
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        # 时间嵌入投影层：将时间向量映射到特征图通道维度
        self.time_proj = nn.Linear(time_dim, out_ch)
        # 第二个3x3卷积
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        # 残差连接（通道数不一致时用1x1卷积对齐，否则恒等映射）
        self.res = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t):
        """
        x: 图像特征 [B, C, H, W]
        t: 时间嵌入 [B, time_dim]
        返回：残差块输出特征
        """
        # 第一次卷积 + SiLU激活
        h = F.silu(self.conv1(x))
        # 时间嵌入融合：先激活 → 投影 → 扩展为 [B, C, 1, 1] → 加到特征图上
        h += self.time_proj(F.silu(t))[:, :, None, None]
        # 第二次卷积 + SiLU激活
        h = F.silu(self.conv2(h))
        # 残差连接：输出 + 原始输入映射
        return h + self.res(x)


# ====================== UNet主体网络 ======================
class UNet(nn.Module):
    """
    DDPM使用的UNet架构
    下采样编码 + 中间瓶颈 + 上采样解码 + 跳跃连接
    """

    def __init__(self, config):
        super().__init__()
        # 基础通道数
        C = config.base_channels
        # 初始化时间嵌入层
        self.time_emb = TimeEmbedding(config.time_emb_dim)
        # 初始卷积层：输入图像 → 基础通道
        self.init = nn.Conv2d(config.in_channels, C, 3, padding=1)

        # ====================== 下采样路径 ======================
        # 第1组残差块 + 下采样
        self.down1 = nn.ModuleList([ResBlock(C, C, config.time_emb_dim),
                                    ResBlock(C, C, config.time_emb_dim)])
        self.pool1 = nn.Conv2d(C, C * 2, 3, 2, 1)  # 通道翻倍，尺寸减半

        # 第2组残差块 + 下采样
        self.down2 = nn.ModuleList([ResBlock(C * 2, C * 2, config.time_emb_dim),
                                    ResBlock(C * 2, C * 2, config.time_emb_dim)])
        self.pool2 = nn.Conv2d(C * 2, C * 4, 3, 2, 1)

        # 第3组残差块 + 下采样
        self.down3 = nn.ModuleList([ResBlock(C * 4, C * 4, config.time_emb_dim),
                                    ResBlock(C * 4, C * 4, config.time_emb_dim)])
        self.pool3 = nn.Conv2d(C * 4, C * 8, 3, 2, 1)

        # 第4组残差块
        self.down4 = nn.ModuleList([ResBlock(C * 8, C * 8, config.time_emb_dim),
                                    ResBlock(C * 8, C * 8, config.time_emb_dim)])

        # ====================== 中间瓶颈层 ======================
        self.mid1 = ResBlock(C * 8, C * 8, config.time_emb_dim)
        self.mid2 = ResBlock(C * 8, C * 8, config.time_emb_dim)

        # ====================== 上采样路径 ======================
        # 反卷积（上采样）层
        self.up4 = nn.ConvTranspose2d(C * 8, C * 4, 4, 2, 1)
        self.up3 = nn.ConvTranspose2d(C * 4, C * 2, 4, 2, 1)
        self.up2 = nn.ConvTranspose2d(C * 2, C, 4, 2, 1)

        # 上采样后的残差块组（包含跳跃连接拼接）
        self.conv4 = nn.ModuleList([ResBlock(C * 8, C * 4, config.time_emb_dim),
                                    ResBlock(C * 4, C * 4, config.time_emb_dim)])
        self.conv3 = nn.ModuleList([ResBlock(C * 4, C * 2, config.time_emb_dim),
                                    ResBlock(C * 2, C * 2, config.time_emb_dim)])
        self.conv2 = nn.ModuleList([ResBlock(C * 2, C, config.time_emb_dim),
                                    ResBlock(C, C, config.time_emb_dim)])

        # 最终输出层：映射回3通道图像
        self.final = nn.Conv2d(C, 3, 3, padding=1)

    def forward(self, x, t):
        """
        UNet前向传播
        x: 输入图像 [B, 3, H, W]
        t: 时间步 [B]
        返回：预测噪声 [B, 3, H, W]
        """
        # 时间步 → 时间嵌入向量
        t = self.time_emb(t)

        # ====================== 下采样编码 ======================
        x1 = self.init(x)
        x1 = self.down1[0](x1, t)
        x1 = self.down1[1](x1, t)

        x2 = self.pool1(x1)
        x2 = self.down2[0](x2, t)
        x2 = self.down2[1](x2, t)

        x3 = self.pool2(x2)
        x3 = self.down3[0](x3, t)
        x3 = self.down3[1](x3, t)

        x4 = self.pool3(x3)
        x4 = self.down4[0](x4, t)
        x4 = self.down4[1](x4, t)

        # ====================== 中间层 ======================
        x4 = self.mid1(x4, t)
        x4 = self.mid2(x4, t)

        # ====================== 上采样解码 + 跳跃连接 ======================
        x = self.up4(x4)
        x = torch.cat([x, x3], dim=1)  # 拼接下采样特征
        x = self.conv4[0](x, t)
        x = self.conv4[1](x, t)

        x = self.up3(x)
        x = torch.cat([x, x2], dim=1)
        x = self.conv3[0](x, t)
        x = self.conv3[1](x, t)

        x = self.up2(x)
        x = torch.cat([x, x1], dim=1)
        x = self.conv2[0](x, t)
        x = self.conv2[1](x, t)

        # 最终输出
        return self.final(x)


# ====================== 测试代码 ======================
if __name__ == "__main__":
    # 创建配置
    config = DDPMConfig()
    # 初始化UNet
    model = UNet(config)
    # 构造测试输入：2张3通道64x64图像
    x = torch.randn(2, 3, 64, 64)
    # 构造测试时间步：2个随机时间步
    t = torch.randint(0, 1000, (2,))
    # 前向推理
    out = model(x, t)
    # 打印维度信息
    print("✅ 输入:", x.shape)
    print("✅ 输出:", out.shape)
    print("✅ 运行成功！")