"""
Tests for gradient_noise atomic feature.
Tests that gradient noise is properly added during training.
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from gpt_lab.train_loops.gradient_noise import run_training
from gpt_lab.train_loops.compiler_validation import SimpleTestTrainingModel, AVAILABLE_DEVICES


@pytest.mark.parametrize("run_training_fn,device", [(run_training, device) for device in AVAILABLE_DEVICES])
def test_gradient_noise_adds_randomness(run_training_fn, device):
    """Test that gradient noise makes training non-deterministic."""
    torch.manual_seed(42)
    
    # Create a simple dataset
    X = torch.randn(64, 8).to(device)
    y = (X.sum(dim=1) > 0).long().to(device)
    ds = TensorDataset(X, y)
    dl = DataLoader(ds, batch_size=16, shuffle=False)
    
    # Create two identical models
    torch.manual_seed(42)
    backbone1 = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 2)).to(device)
    loss_fn = nn.CrossEntropyLoss()
    model1 = SimpleTestTrainingModel(backbone1, loss_fn).to(device)
    optimizer1 = torch.optim.AdamW(model1.parameters(), lr=1e-3)
    
    torch.manual_seed(42)
    backbone2 = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 2)).to(device)
    model2 = SimpleTestTrainingModel(backbone2, loss_fn).to(device)
    optimizer2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
    
    # Run training with gradient noise on model1
    result1 = run_training_fn(
        model=model1,
        optimizer=optimizer1,
        train_loader=dl,
        grad_noise_std=0.01
    )
    
    # Run training without gradient noise on model2 (reset seed to same state)
    torch.manual_seed(42)
    dl2 = DataLoader(ds, batch_size=16, shuffle=False)
    result2 = run_training_fn(
        model=model2,
        optimizer=optimizer2,
        train_loader=dl2,
        grad_noise_std=None
    )
    
    # Verify training completed and result format
    assert isinstance(result1, dict), "Result1 must be a dictionary"
    assert "model" in result1, "Result1 must contain 'model'"
    assert isinstance(result2, dict), "Result2 must be a dictionary"
    assert "model" in result2, "Result2 must contain 'model'"
    
    # Models should have different parameters due to gradient noise
    params1 = list(result1['model'].parameters())
    params2 = list(result2['model'].parameters())
    
    # At least one parameter should be significantly different
    differences_found = False
    for p1, p2 in zip(params1, params2):
        if not torch.allclose(p1.data, p2.data, atol=1e-4, rtol=1e-3):
            differences_found = True
            break
    
    assert differences_found, "Gradient noise should cause parameter differences between models"


# Export the test functions for discovery by catalog_test.py and catalog_llm_compiler.py
__specific_tests__ = [
    test_gradient_noise_adds_randomness,
]
