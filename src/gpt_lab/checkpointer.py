import os
import random
from typing import Optional, Dict, Any
import logging

import numpy as np
import torch
from torch._inductor.virtualized import V
import torch.nn as nn

logger = logging.getLogger(__name__)


def _contains_key(target_key, nested_dict):
    return target_key in nested_dict


def _normalize_state_dict_keys(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Removes the 'module.' and '_orig_mod.' prefix keys from DDP and torch.compile
    """
    state_dict_normalized = {k.replace("module.", "").replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    if state_dict.keys() != state_dict_normalized.keys():
        logger.debug(f"Prefixes from DDP and/or torch.compile removed from state_dict")
    return state_dict_normalized


def save_checkpoint(
    filepath: str,
    metadata: Optional[Dict[str, Any]] = None,
    **stateful_objects: Any,
) -> str:
    """
    Saves a flexible and reproducible training checkpoint.

    Args:
        save_dir: The directory to save the checkpoint in.
        filename: The name of the checkpoint file.
        metadata: Any non-stateful metadata to save (e.g., epoch, step, metrics).
                 Should include info from ReproducibilityManager for full reproducibility.
        **stateful_objects: Keyword arguments for stateful objects to save
            (e.g., model=my_model, optimizer=my_optimizer).

    Returns:
        The full path to the saved checkpoint file.
    """
    logger.info(f"Saving checkpoint to: {filepath}")

    for key in ['git_info', 'rng_state']:
        if not _contains_key('git_info', metadata):
            logger.warning(
                f"Key '{key}' not found in metadata dictionary. "
                f"Future provenance tracking and reproducibility efforts may not be possible without it if the checkpoint file is moved."
            )

    save_dir = os.path.normpath(filepath).split(os.sep)[0]
    os.makedirs(save_dir, exist_ok=True)

    state = {'metadata': metadata}
    for key, obj in stateful_objects.items():
        if hasattr(obj, 'state_dict'):
            logger.debug(f"Adding state_dict for '{key}' to checkpoint...")
            state_dict = obj.state_dict()
            state_dict = _normalize_state_dict_keys(state_dict)
            state[key] = state_dict
            logger.debug(f"Successfully added state_dict for '{key}' to checkpoint")
        else:
            logger.warning(f"Object '{key}' has no .state_dict() method and will not be checkpointed")

    torch.save(state, filepath)
    logger.info(f"Checkpoint saved successfully: {filepath}")
    return filepath


def _value_match_criteria(ckpt_val, trgt_val):
    if type(ckpt_val) != type(trgt_val):
        return False

def _adjust_ckpt_to_target(ckpt_sd: Dict[str, Any], trgt_sd: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalizes state_dict keys to handle inconsistencies from DDP & torch.compile
    when loading a checkpoint's state dictionary.

    Args:
        ckpt_sd: The state dictionary loaded from a checkpoint.
        trgt_sd: The state dictionary loaded from the target object

    Returns:
        The adjusted state dictionary.
    """
    if ckpt_sd == trgt_sd:
        return ckpt_sd
    
    intersect = {
        ckpt_key: ckpt_sd[ckpt_key] 
        for ckpt_key, trgt_key in zip(ckpt_sd, trgt_sd)
        if (ckpt_key in trgt_sd.keys() 
            and trgt_key in ckpt_sd.keys()
            and _value_match_criteria(ckpt_sd[ckpt_key], trgt_sd[trgt_key]))
    }
    only_in_ckpt = ckpt_sd - intersect
    only_in_trgt = trgt_sd - intersect

    ckpt_has_ddp = any(k.find('module.') > -1 for k in ckpt_state_dict.keys())
    target_has_ddp = any(k.find('module.') > -1 for k in target_state_dict.keys())
    ckpt_has_torchcomp = any(k.find('_orig_mod.') > -1 for k in ckpt_state_dict.keys())
    target_has_torchcomp = any(k.find('_orig_mod.') > -1 for k in target_state_dict.keys())

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

    WARNING: This function assumes checkpoint data is safe in order to load non-weight informatino.
    Do not use it to load checkpoints from unknown sources that may contain unsafe data.

    Args:
        filepath: The path to the checkpoint file.
        map_location: The device to map the loaded tensors to ('cpu', 'cuda', etc.).
        **stateful_objects: Keyword arguments for objects to load state into
            (e.g., model=my_model, optimizer=my_optimizer).

    Modifies in-place:
        All stateful objects passed as keyword arguments will have their state
        loaded via their load_state_dict() method if available.

    Returns:
        A dictionary containing all non-state-dict metadata from the checkpoint, if any.
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
                ckpt_state_dict = checkpoint[key]
                ckpt_state_dict = _adjust_ckpt_to_target(ckpt_state_dict, obj.state_dict())

                try:
                    logger.debug(f"Loading state for '{key}'")
                    obj.load_state_dict(ckpt_state_dict)
                except RuntimeError as e:
                    logger.error(
                        f"Failed to load state_dict for '{key}'. This can happen if the "
                        f"architecture does not match the checkpoint."
                    )
                    raise e
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
