"""
Tests for weight_monitoring atomic feature.
Tests that weight monitoring properly tracks weight statistics.
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from utils.device import get_available_devices
from train_loops.catalog.atomic_features.weight_monitoring import run_training


AVAILABLE_DEVICES, _ = get_available_devices()

@pytest.mark.parametrize("run_training_fn,device", [(run_training, device) for device in AVAILABLE_DEVICES])
def test_weight_norm_monitoring(run_training_fn, device):
    """Test that weight norm monitoring works and returns history."""
    torch.manual_seed(42)
    
    # Create a simple dataset with multiple batches
    X = torch.randn(120, 8).to(device)
    y = (X.sum(dim=1) > 0).long().to(device)
    ds = TensorDataset(X, y)
    dl = DataLoader(ds, batch_size=30, shuffle=False)  # 4 batches
    
    # Create model
    model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 2)).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    # Run training with weight norm monitoring
    result = run_training_fn(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        train_loader=dl,
        track_weight_norms=True,
        weight_log_interval=1
    )
    
    # Verify weight norm history is returned
    assert isinstance(result, dict), "Result must be a dictionary"
    assert "weight_norm_history" in result, "Result must contain 'weight_norm_history' when track_weight_norms=True"
    assert isinstance(result["weight_norm_history"], list), "weight_norm_history must be a list"
    assert len(result["weight_norm_history"]) > 0, "weight_norm_history must not be empty"
    
    # Verify all entries are positive numbers
    for norm_val in result["weight_norm_history"]:
        assert isinstance(norm_val, (int, float)), f"Weight norm value must be numeric, got {type(norm_val)}"
        assert norm_val >= 0, f"Weight norm must be non-negative, got {norm_val}"
        assert not torch.isnan(torch.tensor(norm_val)), "Weight norm values must not be NaN"


@pytest.mark.parametrize("run_training_fn,device", [(run_training, device) for device in AVAILABLE_DEVICES])
def test_weight_change_monitoring(run_training_fn, device):
    """Test that weight change monitoring works and returns history."""
    torch.manual_seed(42)
    
    # Create a simple dataset with multiple batches
    X = torch.randn(80, 8).to(device)
    y = (X.sum(dim=1) > 0).long().to(device)
    ds = TensorDataset(X, y)
    dl = DataLoader(ds, batch_size=20, shuffle=False)  # 4 batches
    
    # Create model
    model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 2)).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    # Run training with weight change monitoring
    result = run_training_fn(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        train_loader=dl,
        track_weight_changes=True,
        weight_log_interval=1
    )
    
    # Verify weight change history is returned
    assert isinstance(result, dict), "Result must be a dictionary"
    assert "weight_change_history" in result, "Result must contain 'weight_change_history' when track_weight_changes=True"
    assert isinstance(result["weight_change_history"], list), "weight_change_history must be a list"
    assert len(result["weight_change_history"]) > 0, "weight_change_history must not be empty"
    
    # Verify all entries are non-negative numbers
    for change_val in result["weight_change_history"]:
        assert isinstance(change_val, (int, float)), f"Weight change value must be numeric, got {type(change_val)}"
        assert change_val >= 0, f"Weight change must be non-negative, got {change_val}"
        assert not torch.isnan(torch.tensor(change_val)), "Weight change values must not be NaN"


@pytest.mark.parametrize("run_training_fn,device", [(run_training, device) for device in AVAILABLE_DEVICES])
def test_weight_monitoring_log_interval(run_training_fn, device):
    """Test that weight_log_interval parameter controls monitoring frequency."""
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
    
    # Run training with weight_log_interval=3 (should monitor at batches 0, 3, 6, 9)
    result = run_training_fn(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        train_loader=dl,
        track_weight_norms=True,
        weight_log_interval=3
    )
    
    # Verify weight norm history length respects log_interval
    assert isinstance(result, dict), "Result must be a dictionary"
    assert "weight_norm_history" in result, "Result must contain 'weight_norm_history'"
    
    # With 10 batches and weight_log_interval=3, we expect batches 0, 3, 6, 9 = 4 entries
    expected_entries = 4
    actual_entries = len(result["weight_norm_history"])
    
    assert actual_entries == expected_entries, \
        f"Expected {expected_entries} weight norm entries with weight_log_interval=3, got {actual_entries}"


# Export the test functions for discovery by catalog_test.py and catalog_llm_compiler.py
__specific_tests__ = [
    test_weight_norm_monitoring,
    test_weight_change_monitoring,
    test_weight_monitoring_log_interval
]
