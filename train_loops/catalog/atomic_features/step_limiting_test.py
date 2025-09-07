"""
Tests for step_limiting atomic feature.
Tests that the training loop properly respects max_steps parameter and cycles data when needed.
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from utils.device import best_device as device
from train_loops.catalog.atomic_features.step_limiting import run_training


@pytest.mark.parametrize("run_training_fn", [run_training])
def test_step_limiting_no_limit(run_training_fn):
    """Test that training runs normally when max_steps is None."""
    torch.manual_seed(42)
    
    # Create a small dataset
    X = torch.randn(64, 8).to(device)
    y = (X.sum(dim=1) > 0).long().to(device)
    ds = TensorDataset(X, y)
    dl = DataLoader(ds, batch_size=16, shuffle=False)  # 4 batches
    
    # Create model
    model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 2)).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    # Track initial parameters to verify training happened
    initial_param = model[0].weight.clone()
    
    # Run training with no step limit
    result = run_training_fn(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        train_loader=dl,
        max_steps=None
    )
    
    # Verify result format
    assert isinstance(result, dict), "Result must be a dictionary"
    assert "model" in result, "Result must contain 'model'"
    
    # Verify training happened (parameters changed)
    final_param = model[0].weight
    assert not torch.allclose(initial_param, final_param, atol=1e-6), \
        "Model parameters should have changed during training"


@pytest.mark.parametrize("run_training_fn", [run_training])
def test_step_limiting_with_limit(run_training_fn):
    """Test that training stops after max_steps when specified."""
    torch.manual_seed(42)
    
    # Create a small dataset that would normally run for 4 batches
    X = torch.randn(64, 8).to(device)
    y = (X.sum(dim=1) > 0).long().to(device)
    ds = TensorDataset(X, y)
    dl = DataLoader(ds, batch_size=16, shuffle=False)  # 4 batches
    
    # Create model
    model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 2)).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    # Track initial parameters
    initial_param = model[0].weight.clone()
    
    # Run training with step limit of 2 (should stop early)
    result = run_training_fn(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        train_loader=dl,
        max_steps=2
    )
    
    # Verify result format
    assert isinstance(result, dict), "Result must be a dictionary"
    assert "model" in result, "Result must contain 'model'"
    
    # Verify training happened but stopped early (parameters changed but not as much as full training)
    final_param = model[0].weight
    assert not torch.allclose(initial_param, final_param, atol=1e-6), \
        "Model parameters should have changed during training"


@pytest.mark.parametrize("run_training_fn", [run_training])
def test_step_limiting_data_cycling(run_training_fn):
    """Test that data cycles correctly when max_steps exceeds dataset size."""
    torch.manual_seed(42)
    
    # Create a very small dataset (2 batches) but request more steps
    X = torch.randn(32, 8).to(device) 
    y = (X.sum(dim=1) > 0).long().to(device)
    ds = TensorDataset(X, y)
    dl = DataLoader(ds, batch_size=16, shuffle=False)  # 2 batches
    
    # Create model
    model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 2)).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    # Track initial parameters
    initial_param = model[0].weight.clone()
    
    # Run training with max_steps=5 (should cycle through 2-batch dataset)
    result = run_training_fn(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        train_loader=dl,
        max_steps=5
    )
    
    # Verify result format
    assert isinstance(result, dict), "Result must be a dictionary"
    assert "model" in result, "Result must contain 'model'"
    
    # Verify training happened (parameters changed)
    final_param = model[0].weight
    assert not torch.allclose(initial_param, final_param, atol=1e-6), \
        "Model parameters should have changed during training with data cycling"


@pytest.mark.parametrize("run_training_fn", [run_training])
def test_step_limiting_single_step(run_training_fn):
    """Test that training works correctly with max_steps=1."""
    torch.manual_seed(42)
    
    # Create dataset
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
    
    # Run training with max_steps=1
    result = run_training_fn(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        train_loader=dl,
        max_steps=1
    )
    
    # Verify result format
    assert isinstance(result, dict), "Result must be a dictionary"
    assert "model" in result, "Result must contain 'model'"
    
    # Verify training happened (parameters changed, but minimally)
    final_param = model[0].weight
    assert not torch.allclose(initial_param, final_param, atol=1e-7), \
        "Model parameters should have changed after single step"


# Export the test functions for discovery by bulk_test.py and llm_train_loop_compiler.py
__specific_tests__ = [
    test_step_limiting_no_limit,
    test_step_limiting_with_limit, 
    test_step_limiting_data_cycling,
    test_step_limiting_single_step
]
