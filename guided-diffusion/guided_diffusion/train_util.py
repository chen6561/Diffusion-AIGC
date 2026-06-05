import copy
import functools
import os

import blobfile as bf
import torch as th
import torch
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW
from torchvision.utils import save_image
import numpy as np

from . import dist_util, logger
from .fp16_util import MixedPrecisionTrainer
from .nn import update_ema
from .resample import LossAwareSampler, UniformSampler

# 初始的半精度损失缩放值，用于稳定FP16训练
INITIAL_LOG_LOSS_SCALE = 20.0


class TrainLoop:
    """
    扩散模型训练循环核心类
    负责：前向传播、损失计算、反向传播、参数更新、模型保存、验证图生成
    """
    def __init__(
            self,
            *,
            model,                  #  UNet 模型
            diffusion,              #  扩散过程类
            data,                   #  训练集数据迭代器
            batch_size,             #  总批次大小
            microbatch,             #  微批次（爆显存时拆分用）
            lr,                     #  学习率
            ema_rate,               #  EMA 滑动平均系数
            log_interval,           #  日志打印间隔
            save_interval,          #  模型保存间隔
            resume_checkpoint,      #  恢复训练的 checkpoint 路径
            use_fp16=False,         #  是否使用半精度训练
            fp16_scale_growth=1e-3, #  半精度损失缩放增长率
            schedule_sampler=None,  #  时间步采样器
            weight_decay=0.0,       #  权重衰减
            lr_anneal_steps=0,      #  学习率衰减总步数
            val_data=None,          #  验证集数据迭代器（用于保存效果图）
    ):
        # 基础组件
        self.model = model
        self.diffusion = diffusion
        self.data = data

        # 训练超参数
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps

        # 验证集（用于保存 LR|SR|HR 拼接图）
        self.val_data = val_data

        # 训练步数计数
        self.step = 0                  # 当前轮步数
        self.resume_step = 0           # 恢复训练时的起始步数
        self.global_batch = self.batch_size * 1  # 总批次（单卡=batch_size）

        # CUDA 同步标志
        self.sync_cuda = th.cuda.is_available()

        # 加载模型参数（断点续训）
        self._load_and_sync_parameters()

        # 半精度训练器
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=fp16_scale_growth,
        )

        # 优化器 AdamW
        self.opt = AdamW(
            self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay
        )

        # 如果是恢复训练，加载 EMA 和优化器状态
        if self.resume_step:
            self._load_optimizer_state()
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.mp_trainer.master_params)
                for _ in range(len(self.ema_rate))
            ]

        # 单卡直接使用模型，不使用 DDP
        if th.cuda.is_available():
            self.use_ddp = True
            self.ddp_model = self.model
        else:
            self.use_ddp = False
            self.ddp_model = self.model

    def _load_and_sync_parameters(self):
        """加载断点续训的模型权重"""
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
            self.model.load_state_dict(
                dist_util.load_state_dict(
                    resume_checkpoint, map_location=dist_util.dev()
                )
            )

    def _load_ema_parameters(self, rate):
        """加载 EMA 权重"""
        ema_params = copy.deepcopy(self.mp_trainer.master_params)

        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
            state_dict = dist_util.load_state_dict(
                ema_checkpoint, map_location=dist_util.dev()
            )
            ema_params = self.mp_trainer.state_dict_to_master_params(state_dict)

        return ema_params

    def _load_optimizer_state(self):
        """加载优化器状态"""
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )
            self.opt.load_state_dict(state_dict)

    def save_val_images(self):
        """
        【核心新增功能】
        模型保存时，自动生成并保存验证集效果图
        格式：LR 低分辨率 | SR 超分结果 | HR 高清真值
        保存路径：./val_output/step_xxxxxx.png
        """
        # 如果没有验证集，直接返回
        if self.val_data is None:
            return
        try:
            # 获取一个验证批次
            val_batch, val_kwargs = next(self.val_data)
            val_batch = val_batch.to(dist_util.dev())
            # 把所有条件变量搬到 GPU
            for k in val_kwargs:
                val_kwargs[k] = val_kwargs[k].to(dist_util.dev())

            # 切换模型为评估模式
            self.model.eval()
            with th.no_grad():
                # 扩散模型采样，生成超分图像 SR
                sr = self.diffusion.p_sample_loop(
                    self.model,
                    val_batch.shape,
                    clip_denoised=True,
                    model_kwargs=val_kwargs,
                    progress=False
                )
            # 切回训练模式
            self.model.train()

            # 反归一化：从 [-1,1] → [0,255]
            def denorm(x):
                return ((x + 1) * 127.5).clamp(0, 255).to(th.uint8)

            # 获取 LR、HR 图像
            lr = val_kwargs["low_res"]
            hr = val_batch
            # 把 LR 上采样到 HR 大小，方便拼接
            lr = th.nn.functional.interpolate(lr, size=hr.shape[-2:], mode="bilinear")

            # 三图横向拼接： LR | SR | HR
            cat = th.cat([denorm(lr), denorm(sr), denorm(hr)], dim=-1)
            save_dir = "./val_output"
            os.makedirs(save_dir, exist_ok=True)
            # 以当前步数命名保存
            path = f"{save_dir}/step_{self.step + self.resume_step:06d}.png"
            save_image(cat.float() / 255.0, path, nrow=1)
            print(f"save val image: {path}")
        except Exception as e:
            print(f"val save error: {e}")

    def run_loop(self):
        """
        主训练循环
        不断取数据 → 跑一步训练 → 打印日志 → 保存模型+验证图
        """
        while (
                not self.lr_anneal_steps
                or self.step + self.resume_step < self.lr_anneal_steps
        ):
            # 取一个批次数据
            batch, cond = next(self.data)
            # 跑一步训练（前向+反向+更新）
            self.run_step(batch, cond)

            # 定时打印日志
            if self.step % self.log_interval == 0:
                logger.dumpkvs()

            # 定时保存模型 + 保存验证效果图
            if self.step % self.save_interval == 0:
                self.save()               # 保存模型
                self.save_val_images()    # 保存验证图

                # 测试模式：保存一次就退出
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return
            self.step += 1

        # 训练结束，保存最后一次模型和验证图
        if (self.step - 1) % self.save_interval != 0:
            self.save()
            self.save_val_images()

    def run_step(self, batch, cond):
        """单步训练：前向、反向、优化、EMA、学习率衰减、日志"""
        self.forward_backward(batch, cond)
        took_step = self.mp_trainer.optimize(self.opt)
        if took_step:
            self._update_ema()
        self._anneal_lr()
        self.log_step()

    def forward_backward(self, batch, cond):
        """
        前向传播 + 损失计算 + 反向传播
        支持微批次拆分（爆显存时自动拆分）
        """
        self.mp_trainer.zero_grad()
        for i in range(0, batch.shape[0], self.microbatch):
            # 取微批次
            micro = batch[i: i + self.microbatch].to(dist_util.dev())
            micro_cond = {
                k: v[i: i + self.microbatch].to(dist_util.dev())
                for k, v in cond.items()
            }
            last_batch = (i + self.microbatch) >= batch.shape[0]
            # 采样时间步 t
            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())

            # 定义损失计算函数
            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                micro,
                t,
                model_kwargs=micro_cond,
            )

            # 前向计算损失
            if last_batch or not self.use_ddp:
                losses = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()

            # 如果是损失感知采样，更新采样权重
            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            # 计算最终损失
            loss = (losses["loss"] * weights).mean()
            # 记录损失日志
            log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )
            # 反向传播
            self.mp_trainer.backward(loss)

    def _update_ema(self):
        """更新 EMA 滑动平均参数"""
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.mp_trainer.master_params, rate=rate)

    def _anneal_lr(self):
        """学习率线性衰减"""
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        """记录当前步数与样本数"""
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)

    def save(self):
        """保存模型权重（普通模型 + EMA 模型）"""
        def save_checkpoint(step, params):
            filename = f"model_{int(step):06d}.pt"
            save_path = os.path.join("./results", filename)
            os.makedirs("./results", exist_ok=True)
            torch.save(params, save_path)
            print(f"model saved: {save_path}")

        # 当前总步数
        current_step = self.step + self.resume_step
        # 保存模型
        save_checkpoint(current_step, self.mp_trainer.master_params)
        # 保存所有 EMA 模型
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(current_step, params)


def parse_resume_step_from_filename(filename):
    """从 checkpoint 文件名解析步数，如 model_001000.pt → 1000"""
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    """获取日志目录"""
    return logger.get_dir()


def find_resume_checkpoint():
    """自动寻找最新 checkpoint（可自行扩展）"""
    return None


def find_ema_checkpoint(main_checkpoint, step, rate):
    """寻找对应的 EMA checkpoint"""
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None


def log_loss_dict(diffusion, ts, losses):
    """记录损失字典，按分位数记录更详细的损失分布"""
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)