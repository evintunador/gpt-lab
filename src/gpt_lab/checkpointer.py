import os
import random
from typing import Optional, Dict, Any
import logging

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def save_checkpoint(
    save_dir: str,
    filename: str,
    metadata: Optional[Dict[str, Any]] = None,
    save_rng_state: bool = True,
    **stateful_objects: Any,
) -> str:
    """
    Saves a flexible and reproducible training checkpoint.

    Args:
        save_dir: The directory to save the checkpoint in.
        filename: The name of the checkpoint file.
        metadata: Any non-stateful metadata to save (e.g., epoch, step, metrics).
                 Should include 'git_info' from ReproducibilityManager for full reproducibility.
        save_rng_state: If True, saves the state of torch, numpy, and random RNGs.
        **stateful_objects: Keyword arguments for stateful objects to save
            (e.g., model=my_model, optimizer=my_optimizer).

    Returns:
        The full path to the saved checkpoint file.
    """
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)
    
    logger.info(f"Saving checkpoint to: {filepath}")

    state = {
        'metadata': metadata or {},
    }

    if save_rng_state:
        state['rng_states'] = {
            'torch': torch.get_rng_state(),
            'numpy': np.random.get_state(),
            'random': random.getstate(),
        }

    for key, obj in stateful_objects.items():
        if hasattr(obj, 'state_dict'):
            state_dict_obj = obj
            if isinstance(state_dict_obj, nn.parallel.DistributedDataParallel):
                logger.debug(f"Unwrapping '{key}' (DDP model) before saving state_dict.")
                state_dict_obj = state_dict_obj.module
            if hasattr(state_dict_obj, '_orig_mod'):
                logger.debug(f"Unwrapping '{key}' (compiled model) before saving state_dict.")
                state_dict_obj = state_dict_obj._orig_mod

            state[key] = state_dict_obj.state_dict()
            logger.debug(f"Added state_dict for '{key}' to checkpoint")
        else:
            logger.warning(f"Object '{key}' has no .state_dict() method and will not be checkpointed")

    torch.save(state, filepath)
    logger.info(f"Checkpoint saved successfully: {filepath}")
    return filepath


def _normalize_state_dict_keys(state_dict: Dict[str, Any], model: nn.Module) -> Dict[str, Any]:
    """
    Normalizes state_dict keys to handle inconsistencies from DDP.

    - If loading into a DDP model, it adds the 'module.' prefix to keys if absent.
    - If loading into a non-DDP model, it removes the 'module.' prefix if present.

    Args:
        state_dict: The state dictionary loaded from a checkpoint.
        model: The model instance to load the state into.

    Returns:
        The adjusted state dictionary.
    """
    has_module_prefix = any(k.startswith("module.") for k in state_dict.keys())
    is_ddp_model = isinstance(model, nn.parallel.DistributedDataParallel)

    if is_ddp_model and not has_module_prefix:
        logger.info("Adding 'module.' prefix to state_dict keys for DDP model compatibility.")
        return {"module." + k: v for k, v in state_dict.items()}
    elif not is_ddp_model and has_module_prefix:
        logger.info("Removing 'module.' prefix from state_dict keys for non-DDP model compatibility.")
        return {k[len("module.") :] : v for k, v in state_dict.items()}

    return state_dict


def load_checkpoint(
    filepath: str,
    map_location: str = "cpu",
    **stateful_objects: Any,
) -> Dict[str, Any]:
    """
    Loads a flexible training checkpoint.

    This function can automatically handle common state_dict key mismatches
    that occur when saving/loading models wrapped with `torch.nn.parallel.DistributedDataParallel`
    or optimized with `torch.compile`.

    Args:
        filepath: The path to the checkpoint file.
        map_location: The device to map the loaded tensors to ('cpu', 'cuda', etc.).
        **stateful_objects: Keyword arguments for objects to load state into
            (e.g., model=my_model, optimizer=my_optimizer).

    Modifies in-place:
        All stateful objects passed as keyword arguments will have their state
        loaded via their load_state_dict() method if available.

    Returns:
        A dictionary containing all non-state-dict data from the checkpoint,
        including metadata and RNG states.
    """
    if not os.path.exists(filepath):
        logger.error(f"Checkpoint file not found: {filepath}")
        raise FileNotFoundError(f"Checkpoint file not found: {filepath}")

    logger.info(f"Loading checkpoint from: {filepath}")
    # Set weights_only=False to allow loading of arbitrary python objects
    # like numpy RNG states, which are not considered "safe" by default.
    checkpoint = torch.load(filepath, map_location=map_location, weights_only=False)

    for key, obj in stateful_objects.items():
        if key in checkpoint:
            if hasattr(obj, "load_state_dict"):
                state_to_load = checkpoint[key]

                target_obj = obj
                if isinstance(target_obj, nn.Module):
                    if isinstance(target_obj, nn.parallel.DistributedDataParallel):
                        target_obj = target_obj.module
                    if hasattr(target_obj, '_orig_mod'):
                        target_obj = target_obj._orig_mod
                try:
                    logger.debug(f"Loading state for '{key}'")
                    target_obj.load_state_dict(state_to_load)
                except RuntimeError as e:
                    logger.error(
                        f"Failed to load state_dict for '{key}'. This can happen if the "
                        f"architecture does not match the checkpoint. Error: {e}"
                    )
                    logger.warning(f"Skipping state loading for '{key}'.")
            else:
                logger.warning(f"Object '{key}' has no .load_state_dict() method. Skipping")
        else:
            logger.warning(f"Key '{key}' not found in checkpoint. Skipping")

    returned_data = {}
    stateful_keys = set(stateful_objects.keys())
    for key, value in checkpoint.items():
        if key not in stateful_keys:
            returned_data[key] = value
    
    logger.info(f"Checkpoint loaded successfully from {filepath}")
    if 'metadata' in returned_data:
        logger.debug(f"Checkpoint metadata: {returned_data['metadata']}")
    return returned_data
