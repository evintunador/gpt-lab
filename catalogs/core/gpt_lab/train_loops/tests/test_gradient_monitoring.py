"""
Tests for gradient_monitoring atomic feature.
Tests that gradient monitoring properly tracks gradient statistics.
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from gpt_lab.train_loops.gradient_monitoring import run_training
from gpt_lab.train_loops.tests.test_utils import SimpleTestTrainingModel, AVAILABLE_DEVICES


@pytest.mark.parametrize("run_training_fn,device", [(run_training, device) for device in AVAILABLE_DEVICES])
def test_gradient_norm_monitoring(run_training_fn, device):
    """Test that gradient norm monitoring works and returns history."""
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
    
    # Run training with gradient norm monitoring
    result = run_training_fn(
        model=model,
        optimizer=optimizer, train_loader=dl,
        track_grad_norms=True,
        grad_log_interval=1
    )
    
    # Verify gradient norm history is returned
    assert isinstance(result, dict), "Result must be a dictionary"
    assert "grad_norm_history" in result, "Result must contain 'grad_norm_history' when track_grad_norms=True"
    assert isinstance(result["grad_norm_history"], list), "grad_norm_history must be a list"
    assert len(result["grad_norm_history"]) > 0, "grad_norm_history must not be empty"
    
    # Verify all entries are positive numbers
    for norm_val in result["grad_norm_history"]:
        assert isinstance(norm_val, (int, float)), f"Gradient norm value must be numeric, got {type(norm_val)}"
        assert norm_val >= 0, f"Gradient norm must be non-negative, got {norm_val}"
        assert not torch.isnan(torch.tensor(norm_val)), "Gradient norm values must not be NaN"


@pytest.mark.parametrize("run_training_fn,device", [(run_training, device) for device in AVAILABLE_DEVICES])
def test_gradient_flow_monitoring(run_training_fn, device):
    """Test that gradient flow monitoring works and returns history."""
    torch.manual_seed(42)
    
    # Create a simple dataset with multiple batches
    X = torch.randn(80, 8).to(device)
    y = (X.sum(dim=1) > 0).long().to(device)
    ds = TensorDataset(X, y)
    dl = DataLoader(ds, batch_size=20, shuffle=False)  # 4 batches
    
    # Create model
    backbone = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 2)).to(device)
    loss_fn = nn.CrossEntropyLoss()
    model = SimpleTestTrainingModel(backbone, loss_fn).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    # Run training with gradient flow monitoring
    result = run_training_fn(
        model=model,
        optimizer=optimizer, train_loader=dl,
        track_grad_flow=True,
        grad_log_interval=1
    )
    
    # Verify gradient flow history is returned
    assert isinstance(result, dict), "Result must be a dictionary"
    assert "grad_flow_history" in result, "Result must contain 'grad_flow_history' when track_grad_flow=True"
    assert isinstance(result["grad_flow_history"], list), "grad_flow_history must be a list"
    assert len(result["grad_flow_history"]) > 0, "grad_flow_history must not be empty"
    
    # Verify all entries are dictionaries with parameter names
    for flow_dict in result["grad_flow_history"]:
        assert isinstance(flow_dict, dict), "Each grad flow entry must be a dictionary"
        assert len(flow_dict) > 0, "Grad flow dict must not be empty"
        
        # Check that parameter names are strings and values are non-negative numbers
        for param_name, grad_norm in flow_dict.items():
            assert isinstance(param_name, str), f"Parameter name must be string, got {type(param_name)}"
            assert isinstance(grad_norm, (int, float)), f"Gradient norm must be numeric, got {type(grad_norm)}"
            assert grad_norm >= 0, f"Gradient norm must be non-negative, got {grad_norm}"


@pytest.mark.parametrize("run_training_fn,device", [(run_training, device) for device in AVAILABLE_DEVICES])
def test_gradient_monitoring_log_interval(run_training_fn, device):
    """Test that grad_log_interval parameter controls monitoring frequency."""
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
    
    # Run training with grad_log_interval=3 (should monitor at batches 0, 3, 6, 9)
    result = run_training_fn(
        model=model,
        optimizer=optimizer, train_loader=dl,
        track_grad_norms=True,
        grad_log_interval=3
    )
    
    # Verify gradient norm history length respects log_interval
    assert isinstance(result, dict), "Result must be a dictionary"
    assert "grad_norm_history" in result, "Result must contain 'grad_norm_history'"
    
    # With 10 batches and grad_log_interval=3, we expect batches 0, 3, 6, 9 = 4 entries
    expected_entries = 4
    actual_entries = len(result["grad_norm_history"])
    
    assert actual_entries == expected_entries, \
        f"Expected {expected_entries} gradient norm entries with grad_log_interval=3, got {actual_entries}"


# Export the test functions for discovery by catalog_test.py and catalog_llm_compiler.py
__specific_tests__ = [
    test_gradient_norm_monitoring,
    test_gradient_flow_monitoring,
    test_gradient_monitoring_log_interval
]
