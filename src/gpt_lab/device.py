from typing import List, Tuple, Any, Union
import logging

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def get_default_device() -> torch.device:
    """
    Selects and returns the best available device as a torch.device object.
    Prioritizes CUDA, then MPS, and falls back to CPU.
    """
    if torch.cuda.is_available():
        device = torch.device(torch.cuda.current_device())
        logger.info(f"Selected default device: {device} (CUDA available)")
        return device
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info(f"Selected default device: {device} (MPS/Apple Silicon)")
        return device
    device = torch.device("cpu")
    logger.info(f"Selected default device: {device} (fallback)")
    return device


def to_device(item: Any, device: Union[str, torch.device], pin_memory: bool = False, non_blocking: bool = False) -> Any:
    """
    Recursively moves a complex data structure to the specified device.
    Supports lists, tuples, dicts, tensors, and nn.Modules.
    
    Args:
        item: The item to move to device
        device: Target device
        pin_memory: If True, pin memory for tensors (only applies to CPU tensors moving to CUDA)
        non_blocking: If True, use non-blocking transfer when possible
    """
    if isinstance(item, (list, tuple)):
        return type(item)(to_device(x, device, pin_memory, non_blocking) for x in item)
    if isinstance(item, dict):
        return {k: to_device(v, device, pin_memory, non_blocking) for k, v in item.items()}
    if hasattr(item, "to") and callable(item.to):
        # For tensors, we can use pin_memory and non_blocking
        if isinstance(item, torch.Tensor):
            if pin_memory and item.device.type == 'cpu' and str(device).startswith('cuda'):
                item = item.pin_memory()
            return item.to(device, non_blocking=non_blocking)
        else:
            # For modules and other objects, just use device
            return item.to(device)
    return item


def dtype_str_parse(dtype: str):
    dtype = dtype.lower()
    dtype_dict = {
        'fp32': torch.float32,
        'fp16': torch.float16,
        'bf16': torch.bfloat16,
        'float32': torch.float32,
        'float16': torch.float16,
        'bfloat16': torch.bfloat16,
    }
    return dtype_dict.get(dtype, dtype)


def to_dtype(item: Any, dtype: str | torch.dtype) -> Any:
    """
    Recursively moves a complex data structure to the specified data type.
    Supports lists, tuples, dicts, tensors, and nn.Modules.
    
    Args:
        item: The item to move to device
        dtype: Target dtype
    """
    if isinstance(dtype, str):
        dtype = dtype_str_parse(dtype)
    if isinstance(dtype, str):
        logger.warning(f"Dtype string '{dtype}' could not be parsed; returning item as-is")
        return item
    if isinstance(item, (list, tuple)):
        return type(item)(to_dtype(x, dtype) for x in item)
    if isinstance(item, dict):
        return {k: to_dtype(v, dtype) for k, v in item.items()}
    if hasattr(item, "to") and callable(item.to):
        return item.to(dtype)
    logger.warning(f"Item {item} could not be converted to dtype {dtype}")
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
        device_count = torch.cuda.device_count()
        logger.debug(f"Found {device_count} CUDA device(s)")
        for i in range(device_count):
            device_name = torch.cuda.get_device_name(i)
            available_devices_with_ranks.append(f'cuda:{i}')
            logger.debug(f"  cuda:{i} - {device_name}")

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
        logger.debug("No accelerators found, using CPU")
    
    logger.info(f"Available devices: {', '.join(available_devices)}")
    return available_devices, available_devices_with_ranks