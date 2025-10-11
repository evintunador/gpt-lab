import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from unittest.mock import patch, call

import gpt_lab.checkpointer as checkpointer

from gpt_lab.train_loops.checkpoint_over_steps import run_training
from gpt_lab.train_loops.tests.test_utils import SimpleTestTrainingModel, AVAILABLE_DEVICES


@pytest.mark.parametrize("device", AVAILABLE_DEVICES)
@patch('gpt_lab.checkpointer.save_checkpoint')
def test_checkpointing_over_steps(mock_save_checkpoint, device, tmp_path):
    """Test that checkpointing is triggered correctly every N steps."""
    torch.manual_seed(0)
    # 10 batches of size 2
    X = torch.randn(20, 4).to(device)
    y = torch.randint(0, 2, (20,)).to(device)
    ds = TensorDataset(X, y)
    dl = DataLoader(ds, batch_size=2) # 10 steps total

    backbone = nn.Linear(4, 2).to(device)
    loss_fn = nn.CrossEntropyLoss()
    model = SimpleTestTrainingModel(backbone, loss_fn).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    output_dir = tmp_path
    save_interval = 3
    num_steps = len(dl) # 10

    run_training(
        model=model,
        optimizer=optimizer,
        train_loader=dl,
        save_every_steps=save_interval,
        output_dir=output_dir,
    )

    # It saves at step -1, every `save_interval`, and at the end if the last step is not a multiple of `save_interval`.
    # For 10 steps and interval 3, it saves at steps: -1, 3, 6, 9, 10
    expected_saves = 1 + (num_steps // save_interval) + (1 if num_steps % save_interval != 0 else 0)
    assert mock_save_checkpoint.call_count == expected_saves
    
    expected_calls = [
        call(
            save_dir=str(output_dir / "checkpoints"),
            filename="step_-1.pt",
            metadata={"step": -1, "config": {}},
            model=model,
            optimizer=optimizer,
        ),
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
        call(
            save_dir=str(output_dir / "checkpoints"),
            filename="step_10.pt",
            metadata={"step": 10, "config": {}},
            model=model,
            optimizer=optimizer,
        ),
    ]
    mock_save_checkpoint.assert_has_calls(expected_calls, any_order=False)

# Registry for discovery
__specific_tests__ = [
    test_checkpointing_over_steps,
]
