import torch
import torch.nn as nn
    

best_device = "cpu"
if torch.cuda.is_available():
    best_device = f"cuda:{torch.cuda.current_device()}"
elif torch.backends.mps.is_available():
    best_device = "mps"


def to_device(item, device: str):
    if isinstance(item, (list, tuple)):
        return tuple(to_device(x, device) for x in item)
    if isinstance(item, dict):
        return {k: to_device(v, device) for k, v in item.items()}
    if torch.is_tensor(item):
        return item.to(device)
    if isinstance(item, nn.Module):
        return item.to(device)
    return item