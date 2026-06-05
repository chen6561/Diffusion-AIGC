import argparse
import inspect

from . import gaussian_diffusion as gd
from .respace import SpacedDiffusion, space_timesteps
from .unet import SuperResModel, UNetModel, EncoderUNetModel

# 分类任务默认类别数量（ImageNet），超分任务不使用
NUM_CLASSES = 1000


def diffusion_defaults():
    """
    扩散模型默认超参数
    """
    return dict(
        learn_sigma=False,  # 模型是否预测方差（False=只预测噪声）
        diffusion_steps=1000,  # 扩散总步数 T
        noise_schedule="linear",  # 噪声增长策略
        timestep_respacing="",  # 时间步重排列（用于快速采样）
        use_kl=False,  # 是否使用KL散度损失
        predict_xstart=False,  # 是否直接预测原始图像x0
        rescale_timesteps=False,  # 是否对时间步进行缩放
        rescale_learned_sigmas=False,  # 是否缩放学习得到的方差
    )


def classifier_defaults():
    """
    分类器模型默认参数
    """
    return dict(
        image_size=64,  # 输入图像尺寸
        classifier_use_fp16=False,  # 分类器是否使用半精度
        classifier_width=128,  # 分类器基础通道宽度
        classifier_depth=2,  # 分类器残差块深度
        classifier_attention_resolutions="32,16,8",  # 注意力层作用的分辨率
        classifier_use_scale_shift_norm=True,  # 是否使用Scale+Shift归一化
        classifier_resblock_updown=True,  # 残差块是否用于下采样/上采样
        classifier_pool="attention",  # 池化方式
    )


def model_and_diffusion_defaults():
    """
    生成模型+扩散过程默认参数
    """
    # 模型基础结构参数
    res = dict(
        image_size=64,  # 输入图像尺寸
        num_channels=128,  # 模型基础通道数
        num_res_blocks=2,  # 每个分辨率的残差块数量
        num_heads=4,  # 注意力头数
        num_heads_upsample=-1,  # 上采样注意力头数（-1=同num_heads）
        num_head_channels=-1,  # 每个注意力头的通道数
        attention_resolutions="16,8",  # 注意力作用的分辨率
        channel_mult="",  # 各阶段通道倍数（自动配置）
        dropout=0.0,  # dropout概率
        class_cond=False,  # 是否使用类别条件
        use_checkpoint=False,  # 是否使用梯度检查点节省显存
        use_scale_shift_norm=True,  # 是否使用Scale+Shift归一化
        resblock_updown=False,  # 残差块是否用于下/上采样
        use_fp16=False,  # 是否使用半精度训练
        use_new_attention_order=False,  # 是否使用新版注意力顺序
    )
    # 合并扩散模型默认参数
    res.update(diffusion_defaults())
    return res


def classifier_and_diffusion_defaults():
    """分类器+扩散模型默认参数"""
    res = classifier_defaults()
    res.update(diffusion_defaults())
    return res


def create_model_and_diffusion(
        image_size,  # 输入图像尺寸
        class_cond,  # 是否使用类别条件
        learn_sigma,  # 是否学习方差
        num_channels,  # 模型基础通道数
        num_res_blocks,  # 每个分辨率残差块数
        channel_mult,  # 通道倍数
        num_heads,  # 注意力头数
        num_head_channels,  # 单头通道数
        num_heads_upsample,  # 上采样注意力头数
        attention_resolutions,  # 注意力分辨率
        dropout,  # dropout率
        diffusion_steps,  # 扩散步数
        noise_schedule,  # 噪声策略
        timestep_respacing,  # 时间步重排
        use_kl,  # 是否KL损失
        predict_xstart,  # 是否预测x0
        rescale_timesteps,  # 是否缩放时间步
        rescale_learned_sigmas,  # 是否缩放学习方差
        use_checkpoint,  # 梯度检查点
        use_scale_shift_norm,  # Scale+Shift归一化
        resblock_updown,  # 残差块下/上采样
        use_fp16,  # 半精度
        use_new_attention_order,  # 新版注意力
):
    """创建标准UNet模型+扩散过程"""
    # 创建UNet模型
    model = create_model(
        image_size,
        num_channels,
        num_res_blocks,
        channel_mult=channel_mult,
        learn_sigma=learn_sigma,
        class_cond=class_cond,
        use_checkpoint=use_checkpoint,
        attention_resolutions=attention_resolutions,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        dropout=dropout,
        resblock_updown=resblock_updown,
        use_fp16=use_fp16,
        use_new_attention_order=use_new_attention_order,
    )
    # 创建扩散过程
    diffusion = create_gaussian_diffusion(
        steps=diffusion_steps,
        learn_sigma=learn_sigma,
        noise_schedule=noise_schedule,
        use_kl=use_kl,
        predict_xstart=predict_xstart,
        rescale_timesteps=rescale_timesteps,
        rescale_learned_sigmas=rescale_learned_sigmas,
        timestep_respacing=timestep_respacing,
    )
    return model, diffusion


def create_model(
        image_size,
        num_channels,
        num_res_blocks,
        channel_mult="",
        learn_sigma=False,
        class_cond=False,
        use_checkpoint=False,
        attention_resolutions="16",
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        dropout=0,
        resblock_updown=False,
        use_fp16=False,
        use_new_attention_order=False,
):
    """创建标准UNet图像生成模型"""
    # 如果未指定通道倍数，根据图像尺寸自动设置
    if channel_mult == "":
        if image_size == 512:
            channel_mult = (0.5, 1, 1, 2, 2, 4, 4)
        elif image_size == 256:
            channel_mult = (1, 1, 2, 2, 4, 4)
        elif image_size == 128:
            channel_mult = (1, 1, 2, 3, 4)
        elif image_size == 64:
            channel_mult = (1, 2, 3, 4)
        else:
            raise ValueError(f"unsupported image size: {image_size}")
    else:
        # 把字符串格式的channel_mult转为元组
        channel_mult = tuple(int(ch_mult) for ch_mult in channel_mult.split(","))

    # 存储需要加入注意力机制的特征图尺寸
    attention_ds = []
    for res in attention_resolutions.split(","):
        attention_ds.append(image_size // int(res))

    # 构建并返回UNet模型
    return UNetModel(
        image_size=image_size,  # 模型输入图像尺寸
        in_channels=3,  # 输入通道数（RGB=3）
        model_channels=num_channels,  # 模型基础通道数
        out_channels=(3 if not learn_sigma else 6),  # 输出3或6通道
        num_res_blocks=num_res_blocks,  # 每个分辨率残差块数量
        attention_resolutions=tuple(attention_ds),  # 加入注意力的分辨率
        dropout=dropout,  # dropout率
        channel_mult=channel_mult,  # 各阶段通道倍数
        num_classes=(NUM_CLASSES if class_cond else None),  # 类别数
        use_checkpoint=use_checkpoint,  # 梯度检查点
        use_fp16=use_fp16,  # 半精度
        num_heads=num_heads,  # 注意力头数
        num_head_channels=num_head_channels,  # 单头通道数
        num_heads_upsample=num_heads_upsample,  # 上采样注意力头数
        use_scale_shift_norm=use_scale_shift_norm,  # 归一化方式
        resblock_updown=resblock_updown,  # 残差块下/上采样
        use_new_attention_order=use_new_attention_order,  # 新版注意力
    )


def create_classifier_and_diffusion(
        image_size,
        classifier_use_fp16,
        classifier_width,
        classifier_depth,
        classifier_attention_resolutions,
        classifier_use_scale_shift_norm,
        classifier_resblock_updown,
        classifier_pool,
        learn_sigma,
        diffusion_steps,
        noise_schedule,
        timestep_respacing,
        use_kl,
        predict_xstart,
        rescale_timesteps,
        rescale_learned_sigmas,
):
    """创建分类器模型+扩散过程"""
    # 创建分类器
    classifier = create_classifier(
        image_size,
        classifier_use_fp16,
        classifier_width,
        classifier_depth,
        classifier_attention_resolutions,
        classifier_use_scale_shift_norm,
        classifier_resblock_updown,
        classifier_pool,
    )
    # 创建扩散模型
    diffusion = create_gaussian_diffusion(
        steps=diffusion_steps,
        learn_sigma=learn_sigma,
        noise_schedule=noise_schedule,
        use_kl=use_kl,
        predict_xstart=predict_xstart,
        rescale_timesteps=rescale_timesteps,
        rescale_learned_sigmas=rescale_learned_sigmas,
        timestep_respacing=timestep_respacing,
    )
    return classifier, diffusion


def create_classifier(
        image_size,
        classifier_use_fp16,
        classifier_width,
        classifier_depth,
        classifier_attention_resolutions,
        classifier_use_scale_shift_norm,
        classifier_resblock_updown,
        classifier_pool,
):
    """创建分类器（EncoderUNet）"""
    # 根据图像尺寸自动配置通道倍数
    if image_size == 512:
        channel_mult = (0.5, 1, 1, 2, 2, 4, 4)
    elif image_size == 256:
        channel_mult = (1, 1, 2, 2, 4, 4)
    elif image_size == 128:
        channel_mult = (1, 1, 2, 3, 4)
    elif image_size == 64:
        channel_mult = (1, 2, 3, 4)
    else:
        raise ValueError(f"unsupported image size: {image_size}")

    # 注意力作用的分辨率
    attention_ds = []
    for res in classifier_attention_resolutions.split(","):
        attention_ds.append(image_size // int(res))

    # 返回分类器模型
    return EncoderUNetModel(
        image_size=image_size,  # 输入图像尺寸
        in_channels=3,  # 输入通道
        model_channels=classifier_width,  # 基础通道
        out_channels=1000,  # 输出类别数
        num_res_blocks=classifier_depth,  # 残差块深度
        attention_resolutions=tuple(attention_ds),  # 注意力层
        channel_mult=channel_mult,  # 通道倍数
        use_fp16=classifier_use_fp16,  # 半精度
        num_head_channels=64,  # 注意力头通道
        use_scale_shift_norm=classifier_use_scale_shift_norm,  # 归一化
        resblock_updown=classifier_resblock_updown,  # 下/上采样
        pool=classifier_pool,  # 池化方式
    )


def sr_model_and_diffusion_defaults():
    """超分辨率模型+扩散默认参数"""
    res = model_and_diffusion_defaults()
    res["large_size"] = 256  # 高分辨率图像尺寸
    res["small_size"] = 64  # 低分辨率图像尺寸
    # 只保留超分函数需要的参数
    arg_names = inspect.getfullargspec(sr_create_model_and_diffusion)[0]
    for k in res.copy().keys():
        if k not in arg_names:
            del res[k]
    return res


def sr_create_model_and_diffusion(
        large_size,  # 高分辨率尺寸
        small_size,  # 低分辨率尺寸
        class_cond,  # 是否类别条件
        learn_sigma,  # 是否学习方差
        num_channels,  # 基础通道
        num_res_blocks,  # 残差块数
        num_heads,  # 注意力头
        num_head_channels,  # 单头通道
        num_heads_upsample,  # 上采样注意力头
        attention_resolutions,  # 注意力分辨率
        dropout,  # dropout
        diffusion_steps,  # 扩散步数
        noise_schedule,  # 噪声策略
        timestep_respacing,  # 时间步重排
        use_kl,  # KL损失
        predict_xstart,  # 预测x0
        rescale_timesteps,  # 缩放时间步
        rescale_learned_sigmas,  # 缩放方差
        use_checkpoint,  # 梯度检查点
        use_scale_shift_norm,  # 归一化
        resblock_updown,  # 下/上采样
        use_fp16,  # 半精度
):
    """创建超分模型SuperResModel+扩散过程"""
    # 创建超分模型
    model = sr_create_model(
        large_size,
        small_size,
        num_channels,
        num_res_blocks,
        learn_sigma=learn_sigma,
        class_cond=class_cond,
        use_checkpoint=use_checkpoint,
        attention_resolutions=attention_resolutions,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        dropout=dropout,
        resblock_updown=resblock_updown,
        use_fp16=use_fp16,
    )
    # 创建扩散过程
    diffusion = create_gaussian_diffusion(
        steps=diffusion_steps,
        learn_sigma=learn_sigma,
        noise_schedule=noise_schedule,
        use_kl=use_kl,
        predict_xstart=predict_xstart,
        rescale_timesteps=rescale_timesteps,
        rescale_learned_sigmas=rescale_learned_sigmas,
        timestep_respacing=timestep_respacing,
    )
    return model, diffusion


def sr_create_model(
        large_size,
        small_size,
        num_channels,
        num_res_blocks,
        learn_sigma,
        class_cond,
        use_checkpoint,
        attention_resolutions,
        num_heads,
        num_head_channels,
        num_heads_upsample,
        use_scale_shift_norm,
        dropout,
        resblock_updown,
        use_fp16,
):
    """创建超分辨率专用模型"""
    _ = small_size  # 占位，防止未使用报错

    # 根据高分辨率尺寸设置通道倍数
    if large_size == 512:
        channel_mult = (1, 1, 2, 2, 4, 4)
    elif large_size == 256:
        channel_mult = (1, 1, 2, 2, 4, 4)
    elif large_size == 64:
        channel_mult = (1, 2, 3, 4)
    else:
        raise ValueError(f"unsupported large size: {large_size}")

    # 注意力作用的尺寸
    attention_ds = []
    for res in attention_resolutions.split(","):
        attention_ds.append(large_size // int(res))

    # 构建超分模型并返回
    return SuperResModel(
        image_size=large_size,  # 输出高分辨率尺寸
        in_channels=3,  # 输入通道
        model_channels=num_channels,  # 基础通道
        out_channels=(3 if not learn_sigma else 6),  # 输出通道
        num_res_blocks=num_res_blocks,  # 残差块
        attention_resolutions=tuple(attention_ds),  # 注意力层
        dropout=dropout,  # dropout
        channel_mult=channel_mult,  # 通道倍数
        num_classes=(NUM_CLASSES if class_cond else None),  # 类别条件
        use_checkpoint=use_checkpoint,  # 梯度检查点
        num_heads=num_heads,  # 注意力头
        num_head_channels=num_head_channels,  # 单头通道
        num_heads_upsample=num_heads_upsample,  # 上采样注意力
        use_scale_shift_norm=use_scale_shift_norm,  # 归一化
        resblock_updown=resblock_updown,  # 下/上采样
        use_fp16=use_fp16,  # 半精度
    )


def create_gaussian_diffusion(
        *,
        steps=1000,
        learn_sigma=False,
        sigma_small=False,
        noise_schedule="linear",
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=False,
        rescale_learned_sigmas=False,
        timestep_respacing="",
):
    """创建高斯扩散过程（核心）"""
    # 根据噪声策略生成beta序列
    betas = gd.get_named_beta_schedule(noise_schedule, steps)

    # 确定损失类型
    if use_kl:
        loss_type = gd.LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type = gd.LossType.RESCALED_MSE
    else:
        loss_type = gd.LossType.MSE

    # 如果没有设置重排，则使用完整步数
    if not timestep_respacing:
        timestep_respacing = [steps]

    # 返回支持空间跳跃的扩散模型
    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),  # 使用的时间步
        betas=betas,  # beta序列
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),  # 模型预测目标：噪声 or 原图x0
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),  # 方差类型：固定 or 学习
        loss_type=loss_type,  # 损失函数类型
        rescale_timesteps=rescale_timesteps,  # 是否缩放时间步
    )


def add_dict_to_argparser(parser, default_dict):
    """将参数字典加载到argparse解析器"""
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)


def args_to_dict(args, keys):
    """将命令行参数转为字典格式"""
    return {k: getattr(args, k) for k in keys}


def str2bool(v):
    """
    命令行布尔值解析
    支持：yes/no/true/false/1/0
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")