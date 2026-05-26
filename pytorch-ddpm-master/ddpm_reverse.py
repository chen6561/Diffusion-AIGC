# 解决Windows系统下OpenMP库重复加载导致的冲突问题
# 必须放在导入torch库之前执行，否则无法生效
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 导入PyTorch核心库，用于张量计算与模型推理
import torch
# 导入时间库，用于统计采样耗时
import time
# 导入进度条库，用于可视化采样过程
from tqdm import tqdm
# 导入PIL库，用于生成图像的保存与处理
from PIL import Image
# 导入项目配置类，包含模型参数、训练参数、图像参数等
from config import DDPMConfig
# 导入DDPM模型核心类，包含前向扩散、反向去噪、损失计算等功能
from ddpm import DDPM


def ddpm_backward_sampling(config, checkpoint_path, batch_size=1):
    """
    DDPM 反向采样（生成）函数
    功能：加载训练好的DDPM模型权重，从纯高斯噪声中逐步去噪，生成清晰图像
    参数：
        config: DDPM配置对象，包含步数、图像尺寸、通道数等超参数
        checkpoint_path: 训练好的模型权重文件路径
        batch_size: 单次生成的图像数量，默认为1
    返回：
        samples: 生成好的图像张量
        elapsed_time: 本次采样总耗时
    """
    # ====================== 模型初始化 ======================
    # 按照训练代码的逻辑，创建DDPM模型实例
    # 确保模型结构与训练时完全一致，避免权重不匹配
    ddpm = DDPM(config).to(config.device)

    # 将模型设置为评估模式（推理模式）
    # 关闭Dropout、BatchNorm等训练时特有的层行为
    ddpm.eval()

    # ====================== 加载训练好的模型权重 ======================
    # 加载权重文件，同时指定设备，避免CPU/GPU不匹配
    # weights_only=False 用于兼容PyTorch 2.6+的安全加载机制
    checkpoint = torch.load(checkpoint_path, map_location=config.device, weights_only=False)

    # 将权重加载到模型中
    # 与训练时save_checkpoint保存的结构完全对应
    ddpm.load_state_dict(checkpoint["model_state_dict"])

    # 权重加载完成提示
    print("✅ 模型权重加载成功")

    # ====================== 开始采样计时 ======================
    # 记录采样开始时间戳
    start_time = time.time()

    # ====================== DDPM 反向采样核心流程 ======================
    # torch.no_grad()：禁用梯度计算，节省显存、加快推理速度
    with torch.no_grad():
        # 调用DDPM内置的完整反向采样循环
        # 从T步噪声逐步去噪，生成最终清晰图像
        # samples = ddpm.p_sample_loop(batch_size=batch_size)   # ddpm采样
        samples = ddpm.ddim_sample_loop(batch_size=batch_size)  # ddim采样

    # ====================== 采样结束，计算耗时 ======================
    # 计算从开始到结束的总耗时，保留2位小数
    elapsed_time = time.time() - start_time
    print(f"⏱️ DDPM 采样总耗时：{elapsed_time:.2f} 秒")

    # 返回生成的图像 + 采样耗时
    return samples, elapsed_time


# ====================== 主函数：执行采样流程 ======================
if __name__ == "__main__":
    """
    程序入口
    功能：初始化配置 → 加载模型 → 执行采样 → 保存生成图像
    """
    # 初始化DDPM配置对象，读取所有预设超参数
    config = DDPMConfig()

    # 自动检测设备：优先使用GPU（cuda），无GPU则使用CPU
    config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 训练好的模型权重文件路径（根据实际路径修改）
    checkpoint_path = "./checkpoints/ddpm_epoch_100.pth"

    # ====================== 执行DDPM反向采样 ======================
    # 调用采样函数，获取生成的图像和耗时
    generated_samples, use_time = ddpm_backward_sampling(config, checkpoint_path, batch_size=1)

    # ====================== 保存生成的图像到本地 ======================
    # 创建输出文件夹（不存在则自动创建）
    os.makedirs("output", exist_ok=True)

    # 遍历生成的每一张图像，保存为PNG格式
    for i, img in enumerate(generated_samples):
        # 将张量从 [C, H, W] 转为 [H, W, C]，适配PIL图像格式
        img = img.permute(1, 2, 0).cpu().numpy()

        # 将模型输出范围从 [-1, 1] 转换为图像像素范围 [0, 255]
        img = (img.clip(-1, 1) + 1) / 2 * 255

        # 保存为本地图片
        Image.fromarray(img.astype('uint8')).save(f"output/sample_{i}.png")

    # 完成提示
    print("✅ 采样完成！已保存到 output/ 文件夹")