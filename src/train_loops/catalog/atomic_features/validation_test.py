"""
Tests for validation atomic feature.
Tests that the training loop properly handles validation during training.
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from utils.device import best_device as device
from train_loops.catalog.atomic_features.validation import run_training


@pytest.mark.parametrize("run_training_fn", [run_training])
def test_validation_with_loader(run_training_fn):
    """Test that validation is performed and val_loss_history is returned when val_loader is provided."""
    torch.manual_seed(42)
    
    # Create training dataset
    X_train = torch.randn(128, 16).to(device)
    y_train = (X_train.sum(dim=1) > 0).long().to(device)
    train_ds = TensorDataset(X_train, y_train)
    train_dl = DataLoader(train_ds, batch_size=32, shuffle=False)
    
    # Create validation dataset
    X_val = torch.randn(64, 16).to(device)
    y_val = (X_val.sum(dim=1) > 0).long().to(device)
    val_ds = TensorDataset(X_val, y_val)
    val_dl = DataLoader(val_ds, batch_size=16, shuffle=False)
    
    # Create model
    model = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 2)).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    # Run training with validation
    result = run_training_fn(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        train_loader=train_dl,
        val_loader=val_dl,
        val_interval=2  # Validate every 2 batches
    )
    
    # Verify val_loss_history is returned
    assert isinstance(result, dict), "Result must be a dictionary"
    assert "val_loss_history" in result, "Result must contain 'val_loss_history' when val_loader is provided"
    assert isinstance(result["val_loss_history"], list), "val_loss_history must be a list"
    assert len(result["val_loss_history"]) > 0, "val_loss_history must not be empty"
    
    # Verify all entries are numbers
    for loss_val in result["val_loss_history"]:
        assert isinstance(loss_val, (int, float)), f"Validation loss value must be numeric, got {type(loss_val)}"
        assert not torch.isnan(torch.tensor(loss_val)), "Validation loss values must not be NaN"


@pytest.mark.parametrize("run_training_fn", [run_training])
def test_validation_without_loader(run_training_fn):
    """Test that no val_loss_history is returned when val_loader is not provided."""
    torch.manual_seed(42)
    
    # Create training dataset
    X_train = torch.randn(64, 8).to(device)
    y_train = (X_train.sum(dim=1) > 0).long().to(device)
    train_ds = TensorDataset(X_train, y_train)
    train_dl = DataLoader(train_ds, batch_size=16, shuffle=False)
    
    # Create model
    model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 2)).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    # Run training without validation loader
    result = run_training_fn(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        train_loader=train_dl,
        val_loader=None
    )
    
    # Verify val_loss_history is not returned
    assert isinstance(result, dict), "Result must be a dictionary"
    assert "val_loss_history" not in result, "Result should not contain 'val_loss_history' when val_loader is None"


@pytest.mark.parametrize("run_training_fn", [run_training])
def test_validation_interval(run_training_fn):
    """Test that val_interval parameter controls validation frequency."""
    torch.manual_seed(42)
    
    # Create training dataset with multiple batches
    X_train = torch.randn(300, 8).to(device)
    y_train = (X_train.sum(dim=1) > 0).long().to(device)
    train_ds = TensorDataset(X_train, y_train)
    train_dl = DataLoader(train_ds, batch_size=30, shuffle=False)  # 10 batches
    
    # Create validation dataset
    X_val = torch.randn(60, 8).to(device)
    y_val = (X_val.sum(dim=1) > 0).long().to(device)
    val_ds = TensorDataset(X_val, y_val)
    val_dl = DataLoader(val_ds, batch_size=15, shuffle=False)
    
    # Create model
    model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 2)).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    # Run training with val_interval=3 (should validate at batches 0, 3, 6, 9)
    result = run_training_fn(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        train_loader=train_dl,
        val_loader=val_dl,
        val_interval=3
    )
    
    # Verify val_loss_history length respects val_interval
    assert isinstance(result, dict), "Result must be a dictionary"
    assert "val_loss_history" in result, "Result must contain 'val_loss_history'"
    
    # With 10 batches and val_interval=3, we expect validation at batches 0, 3, 6, 9 = 4 entries
    # Plus potentially the final batch (batch 9 is the last, so it should be included)
    expected_min_entries = 4
    actual_entries = len(result["val_loss_history"])
    
    assert actual_entries >= expected_min_entries, \
        f"Expected at least {expected_min_entries} validation entries with val_interval=3, got {actual_entries}"


# Export the test functions for discovery by catalog_test.py and catalog_llm_compiler.py
__specific_tests__ = [
    test_validation_with_loader,
    test_validation_without_loader,
    test_validation_interval
]
