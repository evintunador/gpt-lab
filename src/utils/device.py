from typing import List, Tuple

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


def get_available_devices(exclude: List[str] = []) -> Tuple[List[str]]:
    available_devices = []
    available_devices_with_ranks = []
    world_size = torch.cuda.device_count()

    # Check for CUDA devices
    if torch.cuda.is_available():
        available_devices.append('cuda')

        if world_size > 1:
            for i in range(world_size):
                available_devices_with_ranks.append(f'cuda:{i}')

    # Check for MPS devices (Apple Silicon)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        available_devices.append('mps')

    # Check for HIP devices (AMD ROCm)
    if hasattr(torch, 'has_hip') and torch.has_hip and torch.device('hip').type == 'hip':
        available_devices.append('hip')

        if world_size > 1:
            for i in range(world_size):
                available_devices_with_ranks.append(f'hip:{i}')

    if exclude:
        def should_keep(dev):
            return not any(ex in dev for ex in exclude)
        available_devices = [dev for dev in available_devices if should_keep(dev)]
        available_devices_with_ranks = [dev for dev in available_devices_with_ranks if should_keep(dev)]

    # Only include CPU if there are no other options
    if not available_devices:
        available_devices.append('cpu')

    return available_devices, available_devices_with_ranks