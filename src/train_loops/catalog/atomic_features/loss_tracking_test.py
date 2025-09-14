"""
Tests for loss_tracking atomic feature.
Tests that the training loop properly tracks and returns loss history.
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from utils.device import get_available_devices
from train_loops.catalog.atomic_features.loss_tracking import run_training


AVAILABLE_DEVICES, _ = get_available_devices()

@pytest.mark.parametrize("run_training_fn,device", [(run_training, device) for device in AVAILABLE_DEVICES])
def test_loss_tracking_returns_history(run_training_fn, device):
    """Test that loss tracking is enabled by default and returns train_loss_history."""
    torch.manual_seed(42)
    
    # Create a simple dataset
    X = torch.randn(128, 16).to(device)
    y = (X.sum(dim=1) > 0).long().to(device)
    ds = TensorDataset(X, y)
    dl = DataLoader(ds, batch_size=32, shuffle=False)
    
    # Create model
    model = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 2)).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    # Run training
    result = run_training_fn(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        train_loader=dl,
        track_loss=True
    )
    
    # Verify train_loss_history is returned
    assert isinstance(result, dict), "Result must be a dictionary"
    assert "train_loss_history" in result, "Result must contain 'train_loss_history' when track_loss=True"
    assert isinstance(result["train_loss_history"], list), "train_loss_history must be a list"
    assert len(result["train_loss_history"]) > 0, "train_loss_history must not be empty"
    
    # Verify all entries are numbers
    for loss_val in result["train_loss_history"]:
        assert isinstance(loss_val, (int, float)), f"Loss value must be numeric, got {type(loss_val)}"
        assert not torch.isnan(torch.tensor(loss_val)), "Loss values must not be NaN"


@pytest.mark.parametrize("run_training_fn,device", [(run_training, device) for device in AVAILABLE_DEVICES])
def test_loss_tracking_disabled(run_training_fn, device):
    """Test that when track_loss=False, no train_loss_history is returned."""
    torch.manual_seed(42)
    
    # Create a simple dataset
    X = torch.randn(64, 8).to(device)
    y = (X.sum(dim=1) > 0).long().to(device)
    ds = TensorDataset(X, y)
    dl = DataLoader(ds, batch_size=16, shuffle=False)
    
    # Create model
    model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 2)).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    # Run training with tracking disabled
    result = run_training_fn(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        train_loader=dl,
        track_loss=False
    )
    
    # Verify train_loss_history is not returned
    assert isinstance(result, dict), "Result must be a dictionary"
    assert "train_loss_history" not in result, "Result should not contain 'train_loss_history' when track_loss=False"


@pytest.mark.parametrize("run_training_fn,device", [(run_training, device) for device in AVAILABLE_DEVICES])
def test_loss_tracking_log_interval(run_training_fn, device):
    """Test that log_interval parameter controls tracking frequency."""
    torch.manual_seed(42)
    
    # Create a dataset with multiple batches
    X = torch.randn(200, 8).to(device)
    y = (X.sum(dim=1) > 0).long().to(device)
    ds = TensorDataset(X, y)
    dl = DataLoader(ds, batch_size=20, shuffle=False)  # 10 batches
    
    # Create model
    model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 2)).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    # Run training with log_interval=3 (should track every 3rd batch: 0, 3, 6, 9)
    result = run_training_fn(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        train_loader=dl,
        track_loss=True,
        log_interval=3
    )
    
    # Verify train_loss_history length respects log_interval
    assert isinstance(result, dict), "Result must be a dictionary"
    assert "train_loss_history" in result, "Result must contain 'train_loss_history'"
    
    # With 10 batches and log_interval=3, we expect batches 0, 3, 6, 9 = 4 entries
    expected_entries = 4
    actual_entries = len(result["train_loss_history"])
    
    assert actual_entries == expected_entries, \
        f"Expected {expected_entries} loss entries with log_interval=3, got {actual_entries}"


# Export the test functions for discovery by catalog_test.py and catalog_llm_compiler.py
__specific_tests__ = [
    test_loss_tracking_returns_history,
    test_loss_tracking_disabled,
    test_loss_tracking_log_interval
]
