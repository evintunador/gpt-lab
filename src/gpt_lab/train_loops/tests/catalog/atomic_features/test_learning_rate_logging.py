"""
Tests for learning_rate_logging atomic feature.
Tests that learning rate logging properly tracks LR changes.
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from gpt_lab.train_loops.learning_rate_logging import run_training
from gpt_lab.train_loops.tests.test_utils import SimpleTestTrainingModel, AVAILABLE_DEVICES


@pytest.mark.parametrize("run_training_fn,device", [(run_training, device) for device in AVAILABLE_DEVICES])
def test_learning_rate_logging_enabled(run_training_fn, device):
    """Test that learning rate logging works and returns history."""
    torch.manual_seed(42)
    
    # Create a simple dataset with multiple batches
    X = torch.randn(120, 8).to(device)
    y = (X.sum(dim=1) > 0).long().to(device)
    ds = TensorDataset(X, y)
    dl = DataLoader(ds, batch_size=30, shuffle=False)  # 4 batches
    
    # Create model
    backbone = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 2)).to(device)
    loss_fn = nn.CrossEntropyLoss()
    model = SimpleTestTrainingModel(backbone, loss_fn).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    # Run training with learning rate logging
    result = run_training_fn(
        model=model,
        optimizer=optimizer, train_loader=dl,
        log_lr_changes=True,
        lr_log_interval=1
    )
    
    # Verify lr history is returned
    assert isinstance(result, dict), "Result must be a dictionary"
    assert "lr_history" in result, "Result must contain 'lr_history' when log_lr_changes=True"
    assert isinstance(result["lr_history"], list), "lr_history must be a list"
    assert len(result["lr_history"]) > 0, "lr_history must not be empty"
    
    # Verify all entries are positive numbers
    for lr_val in result["lr_history"]:
        assert isinstance(lr_val, (int, float)), f"LR value must be numeric, got {type(lr_val)}"
        assert lr_val > 0, f"Learning rate must be positive, got {lr_val}"
        assert not torch.isnan(torch.tensor(lr_val)), "LR values must not be NaN"
    
    # Since we're not changing LR in this test, all values should be the same
    expected_lr = 1e-3
    for lr_val in result["lr_history"]:
        assert abs(lr_val - expected_lr) < 1e-6, f"Expected LR {expected_lr}, got {lr_val}"


@pytest.mark.parametrize("run_training_fn,device", [(run_training, device) for device in AVAILABLE_DEVICES])
def test_learning_rate_logging_log_interval(run_training_fn, device):
    """Test that lr_log_interval parameter controls logging frequency."""
    torch.manual_seed(42)
    
    # Create a dataset with multiple batches
    X = torch.randn(200, 8).to(device)
    y = (X.sum(dim=1) > 0).long().to(device)
    ds = TensorDataset(X, y)
    dl = DataLoader(ds, batch_size=20, shuffle=False)  # 10 batches
    
    # Create model
    backbone = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 2)).to(device)
    loss_fn = nn.CrossEntropyLoss()
    model = SimpleTestTrainingModel(backbone, loss_fn).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    # Run training with lr_log_interval=3 (should log at batches 0, 3, 6, 9)
    result = run_training_fn(
        model=model,
        optimizer=optimizer, train_loader=dl,
        log_lr_changes=True,
        lr_log_interval=3
    )
    
    # Verify LR history length respects log_interval
    assert isinstance(result, dict), "Result must be a dictionary"
    assert "lr_history" in result, "Result must contain 'lr_history'"
    
    # With 10 batches and lr_log_interval=3, we expect batches 0, 3, 6, 9 = 4 entries
    expected_entries = 4
    actual_entries = len(result["lr_history"])
    
    assert actual_entries == expected_entries, \
        f"Expected {expected_entries} LR entries with lr_log_interval=3, got {actual_entries}"


# Export the test functions for discovery by catalog_test.py and catalog_llm_compiler.py
__specific_tests__ = [
    test_learning_rate_logging_enabled,
    test_learning_rate_logging_log_interval
]
