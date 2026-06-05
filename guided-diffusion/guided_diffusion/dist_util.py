import torch

def setup_dist():
    # 不做任何分布式初始化
    pass

def dev():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def rank():
    return 0

def world_size():
    return 1

def sync_params(params):
    pass

def all_reduce(tensor, op=None):
    return tensor

# 新加！修复你当前的报错
def get_world_size():
    return 1

def get_rank():
    return 0