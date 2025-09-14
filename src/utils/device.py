from typing import List, Tuple, Any, Union

import torch
import torch.nn as nn


def get_default_device() -> torch.device:
    """
    Selects and returns the best available device as a torch.device object.
    Prioritizes CUDA, then MPS, and falls back to CPU.
    """
    if torch.cuda.is_available():
        return torch.device(torch.cuda.current_device())
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def to_device(item: Any, device: Union[str, torch.device]) -> Any:
    """
    Recursively moves a complex data structure to the specified device.
    Supports lists, tuples, dicts, tensors, and nn.Modules.
    """
    if isinstance(item, (list, tuple)):
        return type(item)(to_device(x, device) for x in item)
    if isinstance(item, dict):
        return {k: to_device(v, device) for k, v in item.items()}
    # Check for a 'to()' method to handle tensors, modules, etc.
    if hasattr(item, "to") and callable(item.to):
        return item.to(device)
    return item


def get_available_devices(exclude: List[str] = []) -> Tuple[List[str], List[str]]:
    """
    Returns a list of available device types and a list of specific device names.
    Useful for iterating over all available hardware for benchmarking.
    """
    available_devices = []
    available_devices_with_ranks = []

    # Check for CUDA devices
    if torch.cuda.is_available():
        available_devices.append('cuda')
        for i in range(torch.cuda.device_count()):
            available_devices_with_ranks.append(f'cuda:{i}')

    # Check for MPS devices (Apple Silicon)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        available_devices.append('mps')
        available_devices_with_ranks.append('mps')

    # Check for HIP devices (AMD ROCm)
    if hasattr(torch, 'has_hip') and torch.has_hip:
        available_devices.append('hip')
        for i in range(torch.cuda.device_count()):
            available_devices_with_ranks.append(f'hip:{i}')

    if exclude:
        def should_keep(dev):
            return not any(ex in dev for ex in exclude)
        available_devices = [dev for dev in available_devices if should_keep(dev)]
        available_devices_with_ranks = [dev for dev in available_devices_with_ranks if should_keep(dev)]

    # Only include CPU if there are no other options
    if not available_devices:
        available_devices.append('cpu')
        available_devices_with_ranks.append('cpu')

    return available_devices, available_devices_with_ranks