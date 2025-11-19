import os
from typing import Optional, Dict, Any
import logging

import torch

logger = logging.getLogger(__name__)


def _normalize_state_dict_key(
    key: str, 
    substrings: tuple = ("module.", "_orig_mod.")
) -> str:
    """
    Removes all occurrences of any of the given substrings from the key, wherever they appear.

    Args:
        key: The state_dict key to normalize.
        substrings: Tuple of substring strings to remove from anywhere in the key.

    Returns:
        The normalized key.
    """
    for substring in substrings:
        key = key.replace(substring, "")
    return key


def _normalize_state_dict_keys(
    state_dict: Dict[str, Any],
    substrings: tuple = ("module.", "_orig_mod.")
) -> Dict[str, Any]:
    """
    Removes the specified prefix substrings from all keys in the dict, e.g. 'module.' and '_orig_mod.' for DDP and torch.compile.

    Args:
        state_dict: The original state_dict.
        substrings: Tuple of substrings to remove from anywhere in the key.

    Returns:
        The normalized state_dict.
    """
    state_dict_normalized = {_normalize_state_dict_key(k, substrings=substrings): v for k, v in state_dict.items()}
    if list(state_dict.keys()) != list(state_dict_normalized.keys()):
        count_not_equal = sum(
            1 for k1, k2 in zip(state_dict.keys(), state_dict_normalized.keys()) if k1 != k2
        )
        logger.debug(
            f"Prefixes {substrings} removed from state_dict for {count_not_equal} keys."
        )
    return state_dict_normalized


def save_checkpoint(
    filepath: str,
    metadata: Optional[Dict[str, Any]] = None,
    **stateful_objects: Any,
) -> str:
    """
    Saves a flexible and reproducible training checkpoint.

    Args:
        filepath: The directory & name of the checkpoint file.
        metadata: Any non-stateful metadata to save (e.g., epoch, step, metrics).
                 Should include info from ReproducibilityManager for full reproducibility.
        **stateful_objects: Keyword arguments for stateful objects to save
            (e.g., model=my_model, optimizer=my_optimizer).

    Returns:
        The full path to the saved checkpoint file.
    """
    logger.info(f"Saving checkpoint to: {filepath}")

    save_dir = os.path.dirname(filepath)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    state = {'metadata': metadata or {}}
    for key, obj in stateful_objects.items():
        if hasattr(obj, 'state_dict'):
            logger.debug(f"Fetching state_dict of '{key}'.")
            state_dict = obj.state_dict()
            logger.debug(f"Normalizing state_dict of '{key}'.")
            state_dict = _normalize_state_dict_keys(state_dict)
            logger.debug(f"Adding state_dict for '{key}' to checkpoint...")
            state[key] = state_dict
            logger.debug(f"Successfully added state_dict for '{key}' to checkpoint")
        else:
            logger.warning(f"Object '{key}' has no .state_dict() method and will not be checkpointed")

    torch.save(state, filepath)
    logger.info(f"Checkpoint saved successfully: {filepath}")
    return filepath


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
    ckpt_sd_normed = _normalize_state_dict_keys(ckpt_sd)
    
    final_state_dict = trgt_sd.copy()
    matched_ckpt_keys = set()
    for trgt_key in trgt_sd.keys():
        normed_trgt_key = _normalize_state_dict_key(trgt_key)
        if normed_trgt_key in ckpt_sd_normed:
            ckpt_val = ckpt_sd_normed[normed_trgt_key]
            trgt_val = trgt_sd[trgt_key]

            if isinstance(ckpt_val, torch.Tensor) and isinstance(trgt_val, torch.Tensor):
                ckpt_val = ckpt_val.to(device=trgt_val.device, dtype=trgt_val.dtype)

            ckpt_dtype = getattr(ckpt_val, 'dtype', type(ckpt_val))
            trgt_dtype = getattr(trgt_val, 'dtype', type(trgt_val))
            ckpt_shape = getattr(ckpt_val, 'shape', None)
            trgt_shape = getattr(trgt_val, 'shape', None)

            if (ckpt_dtype is not None and trgt_dtype is not None and ckpt_dtype != trgt_dtype) or \
               (ckpt_shape is not None and trgt_shape is not None and ckpt_shape != trgt_shape):
                logger.warning(
                    f"Type/shape mismatch for key '{trgt_key}': "
                    f"checkpoint (dtype={ckpt_dtype}, shape={ckpt_shape}) vs "
                    f"target (dtype={trgt_dtype}, shape={trgt_shape})"
                )

            final_state_dict[trgt_key] = ckpt_val
            matched_ckpt_keys.add(normed_trgt_key)
        else:
            target_val = trgt_sd[trgt_key]
            logger.warning(
                f"Could not find checkpoint value for target key: '{trgt_key}' "
                f"(dtype={getattr(target_val, 'dtype', type(target_val))}, "
                f"shape={getattr(target_val, 'shape', 'N/A')})"
            )

    unused_ckpt_keys = set(ckpt_sd_normed.keys()) - matched_ckpt_keys
    for unused_key in unused_ckpt_keys:
        ckpt_val = ckpt_sd_normed[unused_key]
        logger.warning(
            f"Checkpoint contains unused value for key: '{unused_key}' "
            f"(dtype={getattr(ckpt_val, 'dtype', type(ckpt_val))}, "
            f"shape={getattr(ckpt_val, 'shape', 'N/A')})"
        )

    return final_state_dict


def load_checkpoint(
    filepath: str,
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
    logger.warning(
        "Setting weights_only=False allows loading of arbitrary Python objects from the checkpoint, "
        "including potentially unsafe objects (e.g., numpy RNG states). Only load checkpoints from trusted sources."
    )
    checkpoint = torch.load(filepath, weights_only=False)

    for key, obj in stateful_objects.items():
        if key not in checkpoint:
            logger.warning(f"Object '{key}' not found in checkpoint. Skipping")
            continue
        if not hasattr(obj, "load_state_dict"):
            logger.warning(f"Object '{key}' has no .load_state_dict() method. Skipping")
            continue

        ckpt_obj_sd = checkpoint[key]
        ckpt_obj_sd = _adjust_ckpt_to_target(ckpt_obj_sd, obj.state_dict())

        try:
            logger.debug(f"Loading state for '{key}'")
            obj.load_state_dict(ckpt_obj_sd)
        except RuntimeError as e:
            logger.error(
                f"Failed to load state_dict for '{key}'. This can happen if the "
                f"architecture does not match the checkpoint."
            )
            raise e

    metadata = checkpoint.get('metadata', {})
    if metadata:
        logger.info(f"Checkpoint metadata found.", extra=metadata)
    else:
        logger.info("No checkpoint metadata found.")

    stateful_keys = set(stateful_objects.keys())
    for key, value in checkpoint.items():
        if key not in stateful_keys and key != 'metadata':
            logger.info(f"Found extra checkpoint data under key '{key}'. Adding to metadata dictionary.")
            metadata[key] = value
    
    logger.info(f"Checkpoint loaded successfully from {filepath}")
    return metadata
