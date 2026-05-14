import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from config import DDPMConfig


class ImageDataset(Dataset):
    """自定义图像数据集加载器"""

    def __init__(self, config: DDPMConfig, train: bool = True):
        self.config = config
        self.root = os.path.join(config.dataset_path, "train" if train else "val")
        self.image_paths = [os.path.join(self.root, fname) for fname in os.listdir(self.root)
                            if fname.endswith((".png", ".jpg", ".jpeg"))]

        # 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize((config.image_size, config.image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),  # 转换为[0,1]
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化到[-1,1]
        ])

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        return self.transform(image)


def get_dataloader(config: DDPMConfig, train: bool = True) -> DataLoader:
    """获取数据加载器"""
    dataset = ImageDataset(config, train)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=train,
        num_workers=config.num_workers,
        pin_memory=True
    )