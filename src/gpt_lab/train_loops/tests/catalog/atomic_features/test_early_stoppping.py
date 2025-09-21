"""
Tests for early_stopping atomic feature.
Tests that the training loop properly handles early stopping logic.
Note: Early stopping behavior is hard to test deterministically in a black-box manner,
so these tests focus on parameter handling and basic functionality.
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from gpt_lab.device import get_available_devices
from gpt_lab.train_loops.catalog.atomic_features.early_stopping import run_training


AVAILABLE_DEVICES, _ = get_available_devices()

@pytest.mark.parametrize("run_training_fn,device", [(run_training, device) for device in AVAILABLE_DEVICES])
def test_early_stopping_without_val_loader(run_training_fn, device):
    """Test that training works normally when no validation loader is provided."""
    torch.manual_seed(42)
    
    # Create training dataset
    X = torch.randn(64, 8).to(device)
    y = (X.sum(dim=1) > 0).long().to(device)
    ds = TensorDataset(X, y)
    dl = DataLoader(ds, batch_size=16, shuffle=False)
    
    # Create model
    model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 2)).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    # Track initial parameters
    initial_param = model[0].weight.clone()
    
    # Run training without validation loader (should train normally)
    result = run_training_fn(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        train_loader=dl,
        val_loader=None,
        patience=3,
        min_delta=0.01
    )
    
    # Verify result format
    assert isinstance(result, dict), "Result must be a dictionary"
    assert "model" in result, "Result must contain 'model'"
    
    # Verify training happened (parameters changed)
    final_param = model[0].weight
    assert not torch.allclose(initial_param, final_param, atol=1e-6), \
        "Model parameters should have changed during training"


@pytest.mark.parametrize("run_training_fn,device", [(run_training, device) for device in AVAILABLE_DEVICES])
def test_early_stopping_with_val_loader(run_training_fn, device):
    """Test that early stopping can handle validation loader and parameters."""
    torch.manual_seed(42)
    
    # Create training dataset
    X_train = torch.randn(128, 8).to(device)
    y_train = (X_train.sum(dim=1) > 0).long().to(device)
    train_ds = TensorDataset(X_train, y_train)
    train_dl = DataLoader(train_ds, batch_size=16, shuffle=False)
    
    # Create validation dataset  
    X_val = torch.randn(64, 8).to(device)
    y_val = (X_val.sum(dim=1) > 0).long().to(device)
    val_ds = TensorDataset(X_val, y_val)
    val_dl = DataLoader(val_ds, batch_size=16, shuffle=False)
    
    # Create model
    model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 2)).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    # Track initial parameters
    initial_param = model[0].weight.clone()
    
    # Run training with early stopping parameters
    result = run_training_fn(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        train_loader=train_dl,
        val_loader=val_dl,
        patience=5,
        min_delta=0.01,
        val_interval=2
    )
    
    # Verify result format
    assert isinstance(result, dict), "Result must be a dictionary"
    assert "model" in result, "Result must contain 'model'"
    
    # Verify training happened (parameters changed)
    final_param = model[0].weight
    assert not torch.allclose(initial_param, final_param, atol=1e-6), \
        "Model parameters should have changed during training"


@pytest.mark.parametrize("run_training_fn,device", [(run_training, device) for device in AVAILABLE_DEVICES])
def test_early_stopping_parameter_handling(run_training_fn, device):
    """Test that different early stopping parameters are accepted without errors."""
    torch.manual_seed(42)
    
    # Create small datasets
    X_train = torch.randn(64, 8).to(device)
    y_train = (X_train.sum(dim=1) > 0).long().to(device)
    train_ds = TensorDataset(X_train, y_train)
    train_dl = DataLoader(train_ds, batch_size=16, shuffle=False)
    
    X_val = torch.randn(32, 8).to(device)
    y_val = (X_val.sum(dim=1) > 0).long().to(device)
    val_ds = TensorDataset(X_val, y_val)
    val_dl = DataLoader(val_ds, batch_size=8, shuffle=False)
    
    # Create model
    model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 2)).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    # Test with different parameter combinations
    test_configs = [
        {"patience": 1, "min_delta": 0.0, "val_interval": 1},
        {"patience": 10, "min_delta": 0.1, "val_interval": 5},
        {"patience": 3, "min_delta": 0.001, "val_interval": 2},
    ]
    
    for config in test_configs:
        # Create fresh model for each test
        model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 2)).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        
        # Run training with specific config
        result = run_training_fn(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            train_loader=train_dl,
            val_loader=val_dl,
            **config
        )
        
        # Verify result format
        assert isinstance(result, dict), f"Result must be a dictionary for config {config}"
        assert "model" in result, f"Result must contain 'model' for config {config}"


# Export the test functions for discovery by catalog_test.py and catalog_llm_compiler.py
__specific_tests__ = [
    test_early_stopping_without_val_loader,
    test_early_stopping_with_val_loader,
    test_early_stopping_parameter_handling
]
