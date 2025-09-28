import os

import pytest
import torch
import torch.nn as nn
import torch.optim as optim

from gpt_lab.checkpointer import save_checkpoint, load_checkpoint
from gpt_lab.device import get_default_device
from gpt_lab.reproducibility import get_git_commit_hash


class SimpleModel(nn.Module):
    """A simple model for testing purposes."""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 2)

    def forward(self, x):
        return self.linear(x)


def _checkpointing_roundtrip_test(device: torch.device, tmp_path):
    """
    Core test logic for saving and loading a checkpoint.
    """
    # 1. Setup environment
    save_dir = tmp_path
    filename = "test_checkpoint.pt"
    filepath = os.path.join(save_dir, filename)

    # 2. Create original objects and move to device
    model_orig = SimpleModel().to(device)
    optimizer_orig = optim.Adam(model_orig.parameters(), lr=0.001)
    
    # Create metadata with git info (mimicking what ReproducibilityManager would provide)
    git_info = {
        "commit_hash": get_git_commit_hash(),
        "branch": "test-branch",
        "remote_url": "https://github.com/test/repo.git",
        "was_dirty": False
    }
    metadata_orig = {
        'epoch': 10, 
        'step': 1234, 
        'best_val_loss': 0.05,
        'git_info': git_info
    }

    # Perform a training step to give the optimizer state
    optimizer_orig.zero_grad()
    dummy_input = torch.randn(4, 10, device=device)
    loss = model_orig(dummy_input).sum()
    loss.backward()
    optimizer_orig.step()

    # 3. Save the checkpoint
    save_checkpoint(
        save_dir=str(save_dir),
        filename=filename,
        metadata=metadata_orig,
        save_rng_state=True,
        model=model_orig,
        optimizer=optimizer_orig
    )
    assert os.path.exists(filepath), "Checkpoint file was not created"

    # 4. Create new objects to load into
    model_loaded = SimpleModel().to(device)
    optimizer_loaded = optim.Adam(model_loaded.parameters(), lr=0.001)

    # Verify state is different before loading
    assert not torch.equal(
        next(iter(model_orig.parameters())).data,
        next(iter(model_loaded.parameters())).data
    ), "Models were already identical before loading"

    # 5. Load the checkpoint
    loaded_data = load_checkpoint(
        filepath=filepath,
        map_location=str(device),
        model=model_loaded,
        optimizer=optimizer_loaded
    )

    # 6. Assert that the state has been restored correctly
    assert loaded_data['metadata'] == metadata_orig, "Metadata was not loaded correctly"
    assert 'rng_states' in loaded_data, "RNG states not found in loaded data"
    
    # Verify git info is properly stored in metadata
    assert 'git_info' in loaded_data['metadata'], "Git info not found in metadata"
    assert loaded_data['metadata']['git_info']['commit_hash'] == git_info['commit_hash'], "Git commit hash not preserved"

    # Check model state
    for p_orig, p_loaded in zip(model_orig.parameters(), model_loaded.parameters()):
        assert torch.equal(p_orig.data, p_loaded.data), "Model parameters do not match after loading"

    # Check optimizer state by comparing its components manually.
    # A direct dict comparison fails because it contains tensors.
    sd_orig = optimizer_orig.state_dict()
    sd_loaded = optimizer_loaded.state_dict()
    assert len(sd_orig['param_groups']) == len(sd_loaded['param_groups']), "Number of param groups differs"
    for pg_orig, pg_loaded in zip(sd_orig['param_groups'], sd_loaded['param_groups']):
        # We can't compare 'params' directly as they are just IDs
        pg_orig.pop('params', None)
        pg_loaded.pop('params', None)
        assert pg_orig == pg_loaded, "Optimizer param_groups do not match"

    # Compare state tensors (e.g., momentum buffers)
    assert sd_orig['state'].keys() == sd_loaded['state'].keys(), "Optimizer state keys do not match"
    for param_id in sd_orig['state']:
        state_orig = sd_orig['state'][param_id]
        state_loaded = sd_loaded['state'][param_id]
        for key in state_orig:
            val_orig = state_orig[key]
            val_loaded = state_loaded[key]
            if isinstance(val_orig, torch.Tensor):
                assert torch.equal(val_orig, val_loaded.to(val_orig.device)), f"Optimizer state tensor '{key}' does not match"
            else:
                assert val_orig == val_loaded, f"Optimizer state value '{key}' does not match"


def test_checkpointing_roundtrip(tmp_path):
    """
    Pytest wrapper to run the checkpointing roundtrip test on the default device.
    """
    device = get_default_device()
    _checkpointing_roundtrip_test(device, tmp_path)
