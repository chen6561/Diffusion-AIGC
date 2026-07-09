# SwinIR Multitask Training

这个目录里的 [train_swinir_multitask.py](/C:/Users/Administrator/Documents/Codex/2026-07-02/huggingface-lerobot-e5-86-99-e4/outputs/train_swinir_multitask.py) 是一份面向工业线条超分任务的 PyTorch 训练脚本。

它按下面的方式组织监督：

- 输入：`LR 原图`
- 监督 1：`HR 超分 label`
- 监督 2：`由 HR label 生成的线条 mask`

模型是一个 `SwinIR-M` 风格的共享 backbone，加两个输出头：

- `SR head`：预测高分辨率灰度图
- `Mask head`：预测线条 mask 的 logits

## 1. 数据目录

脚本要求三套目录中的文件名一一对应：

```text
channel_in/
  0001.bmp
  0002.bmp

channel_label/
  0001.bmp
  0002.bmp

channel_mask/
  0001.bmp
  0002.bmp
```

匹配规则很简单：按文件名完全相同配对。

支持的图像后缀：

- `.bmp`
- `.png`
- `.jpg`
- `.jpeg`
- `.tif`
- `.tiff`

## 2. 尺寸要求

- `LR` 图像大小设为 `(H, W)`
- `HR label` 必须是 `(H * scale, W * scale)`
- `mask` 必须和 `HR label` 完全同尺寸

例如 `scale=4` 时：

- `LR`: `128 x 128`
- `HR`: `512 x 512`
- `mask`: `512 x 512`

如果尺寸对不上，脚本会直接报错。

## 3. 训练命令

一个最小启动示例：

```bash
python outputs/train_swinir_multitask.py \
  --train-lr-dir "D:/datasets/东莞算法/超分数据集/channel_in" \
  --train-hr-dir "D:/datasets/东莞算法/超分数据集/channel_label" \
  --train-mask-dir "D:/datasets/东莞算法/超分数据集/channel_mask" \
  --output-dir "./outputs/swinir_multitask_run" \
  --scale 4 \
  --patch-size-lr 64 \
  --batch-size 4 \
  --epochs 200 \
  --amp
```

带验证集的示例：

```bash
python outputs/train_swinir_multitask.py \
  --train-lr-dir "D:/datasets/.../train/channel_in" \
  --train-hr-dir "D:/datasets/.../train/channel_label" \
  --train-mask-dir "D:/datasets/.../train/channel_mask" \
  --val-lr-dir "D:/datasets/.../val/channel_in" \
  --val-hr-dir "D:/datasets/.../val/channel_label" \
  --val-mask-dir "D:/datasets/.../val/channel_mask" \
  --output-dir "./outputs/swinir_multitask_run" \
  --scale 4 \
  --batch-size 4 \
  --epochs 200 \
  --amp
```

## 4. 默认损失

脚本里默认总损失是：

```text
L_total =
  1.0 * L_sr
+ 0.5 * L_mask
+ 0.2 * L_edge
+ 0.2 * L_topo
```

具体对应：

- `L_sr`: `CharbonnierLoss`
- `L_mask`: `BCEWithLogits + Dice`
- `L_edge`: `Sobel edge L1`
- `L_topo`: `Soft clDice`

这套组合更偏向“线条稳定”，而不是 GAN 那种“看起来更锐”。

## 5. 依赖

至少需要：

```bash
pip install torch torchvision pillow numpy
```

如果想启用 GPU 混合精度：

- CUDA 环境要正常
- 启动时加 `--amp`

## 6. 输出内容

训练输出目录里会生成：

- `config.json`: 当前训练配置
- `history.json`: 每个 epoch 的训练和验证指标
- `best_model.pth`: 验证集最优模型
- `checkpoint_epoch_xxxx.pth`: 周期性 checkpoint

## 7. 关键参数

常用参数说明：

- `--scale`: 超分倍率，比如 `2`、`4`
- `--patch-size-lr`: LR patch 大小，HR patch 会自动乘以 `scale`
- `--batch-size`: 批大小
- `--epochs`: 训练轮数
- `--lr`: 学习率
- `--amp`: 使用混合精度
- `--sr-loss-weight`: SR 重建损失权重
- `--mask-loss-weight`: mask 分割损失权重
- `--edge-loss-weight`: 边缘损失权重
- `--topo-loss-weight`: 拓扑损失权重

## 8. 推荐起步配置

如果你的目标是“最内层正极线条更稳定”，建议先从这一版开始：

```text
scale = 4
patch_size_lr = 64
batch_size = 4
lr = 2e-4
sr_loss_weight = 1.0
mask_loss_weight = 0.5
edge_loss_weight = 0.2
topo_loss_weight = 0.2
```

如果发现：

- 线条位置还会飘：增加 `mask_loss_weight` 或 `topo_loss_weight`
- 图像发糊：适当提高 `sr_loss_weight`
- 线条边缘断裂：适当提高 `edge_loss_weight`

## 9. 当前脚本的默认假设

这份脚本默认做了几个假设：

- 所有图像都是单通道灰度图
- `mask` 是二值图，读取时按 `127` 阈值转成 `0/1`
- 训练时按随机裁块和翻转增强
- 验证时不裁块，直接整图推理

如果你的 `mask` 不是严格二值，或者想做多类别分割，需要再改一下读取和损失部分。

## 10. 建议的下一步

这份脚本已经够用来做第一轮验证。更进一步时，比较值得加的东西有：

- 自动生成 train/val 文件列表
- 训练中保存可视化样本
- 增加 `Focal Loss` 选项
- 增加从现有 ESRGAN/RRDB 权重迁移初始化的入口
- 将 mask 分成 `thick mask` 和 `skeleton mask` 两种监督
