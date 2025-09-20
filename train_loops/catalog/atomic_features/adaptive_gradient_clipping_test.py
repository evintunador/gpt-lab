"""
Tests for adaptive_gradient_clipping atomic feature.
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from device import get_available_devices
from .adaptive_gradient_clipping import run_training


AVAILABLE_DEVICES, _ = get_available_devices()

@pytest.mark.parametrize("run_training_fn,device", [(run_training, device) for device in AVAILABLE_DEVICES])
def test_adaptive_gradient_clipping_applied(run_training_fn, device):
    """Test that adaptive gradient clipping affects training when gradients are large."""
    torch.manual_seed(42)
    
    # Create a simple dataset with large values to induce large gradients
    X = torch.randn(64, 8).to(device) * 10  # Scale up inputs
    y = (X.sum(dim=1) > 0).long().to(device)
    ds = TensorDataset(X, y)
    dl = DataLoader(ds, batch_size=16, shuffle=False)
    
    # Create two identical models
    torch.manual_seed(42)
    model1 = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 2)).to(device)
    optimizer1 = torch.optim.SGD(model1.parameters(), lr=1e-1)  # High LR to induce large gradients
    
    torch.manual_seed(42)
    model2 = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 2)).to(device)
    optimizer2 = torch.optim.SGD(model2.parameters(), lr=1e-1)
    
    loss_fn = nn.CrossEntropyLoss()
    
    # Run training with AGC
    result1 = run_training_fn(
        model=model1,
        optimizer=optimizer1,
        loss_fn=loss_fn,
        train_loader=dl,
        agc_clip_factor=0.01  # Small clip factor to trigger clipping
    )
    
    # Run training without AGC
    dl2 = DataLoader(ds, batch_size=16, shuffle=False)
    result2 = run_training_fn(
        model=model2,
        optimizer=optimizer2,
        loss_fn=loss_fn,
        train_loader=dl2,
        agc_clip_factor=None
    )
    
    # Verify training completed and result format
    assert isinstance(result1, dict), "Result1 must be a dictionary"
    assert "model" in result1, "Result1 must contain 'model'"
    assert isinstance(result2, dict), "Result2 must be a dictionary"
    assert "model" in result2, "Result2 must contain 'model'"
    
    # Models should have different parameters due to gradient clipping
    params1 = list(result1['model'].parameters())
    params2 = list(result2['model'].parameters())
    
    # At least one parameter should be different
    differences_found = False
    for p1, p2 in zip(params1, params2):
        if not torch.allclose(p1.data, p2.data, atol=1e-4, rtol=1e-3):
            differences_found = True
            break
    
    assert differences_found, "AGC should cause parameter differences between models"


# Export the test functions for discovery by catalog_test.py and catalog_llm_compiler.py
__specific_tests__ = [
    test_adaptive_gradient_clipping_applied,
]
