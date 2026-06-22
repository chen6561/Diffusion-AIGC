import os
import random
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import torch
import yaml


@dataclass
class AverageMeter:
    value: float = 0.0
    sum: float = 0.0
    count: int = 0

    @property
    def avg(self) -> float:
        return 0.0 if self.count == 0 else self.sum / self.count

    def update(self, value: float, n: int = 1) -> None:
        self.value = value
        self.sum += value * n
        self.count += n


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_checkpoint(state: Dict[str, Any], path: str) -> None:
    ensure_dir(os.path.dirname(path))
    torch.save(state, path)


def load_checkpoint(path: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer = None) -> int:
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return int(checkpoint.get("epoch", 0))
