from abc import ABC, abstractmethod
import numpy as np
import torch as th
import torch.distributed as dist


def create_named_schedule_sampler(name, diffusion):
    """
    根据名称创建对应的时间步采样器（工厂函数）
    用于在扩散训练中对时间步 t 进行加权采样，降低训练方差

    :param name: 采样器名称，支持 "uniform" / "loss-second-moment"
    :param diffusion: 扩散模型对象
    :return: 实例化的采样器对象
    """
    if name == "uniform":
        # 均匀采样：每个时间步被采样概率相同
        return UniformSampler(diffusion)
    elif name == "loss-second-moment":
        # 损失感知采样：根据历史损失二阶矩动态调整采样概率
        return LossSecondMomentResampler(diffusion)
    else:
        raise NotImplementedError(f"unknown schedule sampler: {name}")


class ScheduleSampler(ABC):
    """
    时间步采样器抽象基类
    作用：对扩散过程的时间步 t 构建一个概率分布，用于重要性采样，降低目标函数方差

    默认情况下：重要性采样无偏，不改变目标函数均值
    子类可覆盖 sample() 方法改变加权方式
    """

    @abstractmethod
    def weights(self):
        """
        获取每个时间步的权重数组（numpy数组）
        权重不需要归一化，但必须为正值
        """

    def sample(self, batch_size, device):
        """
        按权重重要性采样一批时间步 t

        :param batch_size: 采样数量
        :param device: 存放结果的设备（CPU/GPU）
        :return: (timesteps, weights)
                 - timesteps: 采样出的时间步索引
                 - weights: 对应损失加权值，用于校正重要性采样
        """
        # 获取所有时间步的权重
        w = self.weights()
        # 归一化权重 → 概率分布
        p = w / np.sum(w)
        # 根据概率 p 随机采样 batch_size 个时间步
        indices_np = np.random.choice(len(p), size=(batch_size,), p=p)
        # 转 torch tensor
        indices = th.from_numpy(indices_np).long().to(device)

        # 重要性采样权重校正：1/(N*p(t))
        weights_np = 1 / (len(p) * p[indices_np])
        weights = th.from_numpy(weights_np).float().to(device)

        return indices, weights


class UniformSampler(ScheduleSampler):
    """
    均匀时间步采样器（默认）
    每个时间步 t 被采样概率完全相同
    """
    def __init__(self, diffusion):
        # 保存扩散对象
        self.diffusion = diffusion
        # 所有权重初始化为 1 → 均匀分布
        self._weights = np.ones([diffusion.num_timesteps])

    def weights(self):
        """返回均匀权重"""
        return self._weights


class LossAwareSampler(ScheduleSampler):
    """
    损失感知型采样器抽象类
    可根据模型实时损失动态调整采样权重
    支持多卡分布式训练同步
    """

    def update_with_local_losses(self, local_ts, local_losses):
        """
        用当前卡的损失值更新采样权重（会自动多卡同步）

        :param local_ts: 当前卡的时间步张量
        :param local_losses: 当前卡的损失张量
        """
        # 初始化各卡batch大小存储
        batch_sizes = [
            th.tensor([0], dtype=th.int32, device=local_ts.device)
            for _ in range(dist.get_world_size())
        ]
        # 全局聚合各卡batch大小
        dist.all_gather(
            batch_sizes,
            th.tensor([len(local_ts)], dtype=th.int32, device=local_ts.device),
        )

        # 获取所有卡batch大小，并取最大值用于对齐填充
        batch_sizes = [x.item() for x in batch_sizes]
        max_bs = max(batch_sizes)

        # 初始化全局时间步、损失缓冲区
        timestep_batches = [th.zeros(max_bs).to(local_ts) for bs in batch_sizes]
        loss_batches = [th.zeros(max_bs).to(local_losses) for bs in batch_sizes]

        # 全局聚合所有卡的时间步与损失
        dist.all_gather(timestep_batches, local_ts)
        dist.all_gather(loss_batches, local_losses)

        # 展平所有有效数据
        timesteps = [
            x.item() for y, bs in zip(timestep_batches, batch_sizes) for x in y[:bs]
        ]
        losses = [
            x.item() for y, bs in zip(loss_batches, batch_sizes) for x in y[:bs]
        ]

        # 使用全局统一的时间步与损失更新采样器
        self.update_with_all_losses(timesteps, losses)

    @abstractmethod
    def update_with_all_losses(self, ts, losses):
        """
        由所有卡的损失更新权重（子类必须实现）
        """


class LossSecondMomentResampler(LossAwareSampler):
    """
    基于损失二阶矩的动态采样器
    权重 ∝ sqrt( E[ loss^2 ] )
    让模型更关注损失大、难学习的时间步
    """
    def __init__(self, diffusion, history_per_term=10, uniform_prob=0.001):
        self.diffusion = diffusion                  # 扩散模型
        self.history_per_term = history_per_term    # 每个时间步保存的历史损失数量
        self.uniform_prob = uniform_prob            # 保留均匀采样概率，防止崩溃

        # 损失历史记录：[num_timesteps, history_per_term]
        self._loss_history = np.zeros(
            [diffusion.num_timesteps, history_per_term], dtype=np.float64
        )
        # 每个时间步已记录的损失数量
        self._loss_counts = np.zeros([diffusion.num_timesteps], dtype=np.int)

    def weights(self):
        """
        计算每个时间步的采样权重
        如果未预热完成 → 返回均匀权重
        否则 → 权重 = sqrt(平均损失平方)
        """
        # 未预热：所有时间步损失未收集足够历史
        if not self._warmed_up():
            return np.ones([self.diffusion.num_timesteps], dtype=np.float64)

        # 用损失二阶矩构造权重
        weights = np.sqrt(np.mean(self._loss_history ** 2, axis=-1))
        # 归一化
        weights /= np.sum(weights)
        # 混合一部分均匀概率，避免某些时间步永远不被采样
        weights *= 1 - self.uniform_prob
        weights += self.uniform_prob / len(weights)
        return weights

    def update_with_all_losses(self, ts, losses):
        """
        用全局时间步和损失更新历史损失记录
        每个时间步最多保留 history_per_term 个最新损失
        """
        for t, loss in zip(ts, losses):
            if self._loss_counts[t] == self.history_per_term:
                # 损失记录已满，移出最旧的，加入最新的
                self._loss_history[t, :-1] = self._loss_history[t, 1:]
                self._loss_history[t, -1] = loss
            else:
                # 记录未满，直接追加
                self._loss_history[t, self._loss_counts[t]] = loss
                self._loss_counts[t] += 1

    def _warmed_up(self):
        """检查是否所有时间步都收集满了历史损失"""
        return (self._loss_counts == self.history_per_term).all()