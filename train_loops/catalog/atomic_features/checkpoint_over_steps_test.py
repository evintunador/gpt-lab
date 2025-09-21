import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from unittest.mock import patch, call

import os
import checkpointer

from device import get_available_devices
from .checkpoint_over_steps import run_training

AVAILABLE_DEVICES, _ = get_available_devices()

@pytest.mark.parametrize("device", AVAILABLE_DEVICES)
@patch('train_loops.catalog.atomic_features.checkpoint_over_steps.checkpointer.save_checkpoint')
def test_checkpointing_over_steps(mock_save_checkpoint, device, tmp_path):
    """Test that checkpointing is triggered correctly every N steps."""
    torch.manual_seed(0)
    # 10 batches of size 2
    X = torch.randn(20, 4).to(device)
    y = torch.randint(0, 2, (20,)).to(device)
    ds = TensorDataset(X, y)
    dl = DataLoader(ds, batch_size=2) # 10 steps total

    model = nn.Linear(4, 2).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()

    output_dir = tmp_path
    save_interval = 3
    num_steps = len(dl) # 10

    run_training(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        train_loader=dl,
        save_every_steps=save_interval,
        output_dir=output_dir,
    )

    # Check that save_checkpoint was called at steps 3, 6, 9
    assert mock_save_checkpoint.call_count == num_steps // save_interval # 10 // 3 = 3
    
    expected_calls = [
        call(
            save_dir=str(output_dir / "checkpoints"),
            filename="step_3.pt",
            metadata={"step": 3, "config": {}},
            model=model,
            optimizer=optimizer,
        ),
        call(
            save_dir=str(output_dir / "checkpoints"),
            filename="step_6.pt",
            metadata={"step": 6, "config": {}},
            model=model,
            optimizer=optimizer,
        ),
        call(
            save_dir=str(output_dir / "checkpoints"),
            filename="step_9.pt",
            metadata={"step": 9, "config": {}},
            model=model,
            optimizer=optimizer,
        ),
    ]
    mock_save_checkpoint.assert_has_calls(expected_calls, any_order=False)

# Registry for discovery
__specific_tests__ = [
    test_checkpointing_over_steps,
]
