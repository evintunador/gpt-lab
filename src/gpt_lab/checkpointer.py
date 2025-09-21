import os
import random
from typing import Optional, Dict, Any

import numpy as np
import torch


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

    state = {
        'metadata': metadata or {},
    }

    if save_rng_state:
        state['rng_states'] = {
            'torch': torch.get_rng_state(),
            'numpy': np.random.get_state(),
            'random': random.getstate(),
        }

    # Save state dict for any object that has one
    for key, obj in stateful_objects.items():
        if hasattr(obj, 'state_dict'):
            state[key] = obj.state_dict()
        else:
            print(f"Warning: object '{key}' has no .state_dict() method and will not be checkpointed.")

    torch.save(state, filepath)
    print(f"Checkpoint saved to {filepath}")
    return filepath


def load_checkpoint(
    filepath: str,
    map_location: str = 'cpu',
    **stateful_objects: Any,
) -> Dict[str, Any]:
    """
    Loads a flexible training checkpoint.

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
        raise FileNotFoundError(f"Checkpoint file not found: {filepath}")

    # Set weights_only=False to allow loading of arbitrary python objects
    # like numpy RNG states, which are not considered "safe" by default.
    checkpoint = torch.load(filepath, map_location=map_location, weights_only=False)
    
    # Load state into the provided objects
    for key, obj in stateful_objects.items():
        if key in checkpoint:
            if hasattr(obj, 'load_state_dict'):
                print(f"Loading state for '{key}'...")
                obj.load_state_dict(checkpoint[key])
            else:
                print(f"Warning: object '{key}' has no .load_state_dict() method. Skipping.")
        else:
            print(f"Warning: key '{key}' not found in checkpoint. Skipping.")

    # Return all data that isn't a state_dict for the passed objects
    returned_data = {}
    stateful_keys = set(stateful_objects.keys())
    for key, value in checkpoint.items():
        if key not in stateful_keys:
            returned_data[key] = value
    
    print(f"Loaded checkpoint from {filepath}")
    return returned_data
