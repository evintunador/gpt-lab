"""
Tests for the device atomic feature.
Ensures that the model and all data batches are correctly moved to the target device.
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from gpt_lab.train_loops.device import run_training
from gpt_lab.train_loops.tests.test_utils import SimpleTestTrainingModel, AVAILABLE_DEVICES


# AVAILABLE_DEVICES doesn't include 'cpu' funnily enough
@pytest.mark.parametrize("run_training_fn,device", [(run_training, device) for device in AVAILABLE_DEVICES + ['cpu']])
def test_device_placement_correction(run_training_fn, device):
    """
    Tests that the training loop moves a model and data from a mismatched
    device to the correct target device.
    """
    target_device = device
    
    # Smarter mismatch logic to avoid the 'meta' device issue.
    if target_device == "cpu":
        # Find another available device to use as the mismatch source.
        other_devices = [d for d in AVAILABLE_DEVICES if d != "cpu"]
        if not other_devices:
            pytest.skip("Cannot test CPU placement without another device to mismatch from.")
        mismatch_device = other_devices[0]
    else:
        mismatch_device = "cpu"

    # 1. Create model and data on the WRONG device
    torch.manual_seed(42)
    backbone = nn.Sequential(nn.Linear(16, 8), nn.ReLU(), nn.Linear(8, 2)).to(mismatch_device)
    model = SimpleTestTrainingModel(backbone, nn.CrossEntropyLoss()).to(mismatch_device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    X_train = torch.randn(64, 16).to(mismatch_device)
    y_train = torch.randint(0, 2, (64,)).to(mismatch_device)
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=16)

    # Pre-condition check: model parameters are on the mismatch_device
    assert all(p.device.type == mismatch_device.split(':')[0] for p in model.parameters())

    # 2. Run training and instruct it to move everything to the TARGET device
    try:
        result = run_training_fn(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            device=target_device,
        )
    except RuntimeError as e:
        pytest.fail(
            f"Training loop failed with a RuntimeError, likely due to a device mismatch. "
            f"Error: {e}"
        )

    # 3. Assert that the model is now on the CORRECT device
    assert "model" in result, "Result dictionary must contain the trained model."
    final_model = result["model"]
    
    for p in final_model.parameters():
        assert p.device.type == target_device.split(':')[0], \
            f"Model parameter was not moved to target device. Expected {target_device}, found {p.device}."


# Export the test functions for discovery by the compiler
__specific_tests__ = [
    test_device_placement_correction,
]
