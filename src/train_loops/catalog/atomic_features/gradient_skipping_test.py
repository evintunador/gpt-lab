"""
Tests for gradient_skipping atomic feature.
Tests that gradient skipping properly handles problematic gradients.
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from utils.device import get_available_devices
from train_loops.catalog.atomic_features.gradient_skipping import run_training


AVAILABLE_DEVICES, _ = get_available_devices()

@pytest.mark.parametrize("run_training_fn,device", [(run_training, device) for device in AVAILABLE_DEVICES])
def test_gradient_skipping_norm_threshold(run_training_fn, device):
    """Test that gradient skipping based on norm threshold works."""
    torch.manual_seed(42)
    
    # Create a simple dataset with large values to potentially induce large gradients
    X = torch.randn(64, 8).to(device) * 10  # Scale up to potentially trigger skipping
    y = (X.sum(dim=1) > 0).long().to(device)
    ds = TensorDataset(X, y)
    dl = DataLoader(ds, batch_size=16, shuffle=False)
    
    # Create model
    model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 2)).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)  # High LR to potentially trigger large gradients
    
    # Run training with gradient skipping enabled
    result = run_training_fn(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        train_loader=dl,
        grad_skip_threshold=0.1,  # Very low threshold to trigger skipping
        skip_strategy="norm"
    )
    
    # Verify skipping statistics are returned
    assert isinstance(result, dict), "Result must be a dictionary"
    assert "skipped_steps" in result, "Result must contain 'skipped_steps' when grad_skip_threshold is set"
    assert "total_steps" in result, "Result must contain 'total_steps' when grad_skip_threshold is set"
    
    assert isinstance(result["skipped_steps"], int), "skipped_steps must be an integer"
    assert isinstance(result["total_steps"], int), "total_steps must be an integer"
    assert result["skipped_steps"] >= 0, "skipped_steps must be non-negative"
    assert result["total_steps"] > 0, "total_steps must be positive"
    assert result["skipped_steps"] <= result["total_steps"], "skipped_steps cannot exceed total_steps"


# Export the test functions for discovery by catalog_test.py and catalog_llm_compiler.py
__specific_tests__ = [
    test_gradient_skipping_norm_threshold,
]
