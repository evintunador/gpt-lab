"""
Comprehensive tests for smart_api.py functionality.
Tests the visual validations as proper unit tests.
"""

from typing import Dict, Any
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pytest

from utils.device import best_device as device
from train_loops.smart_api import smart_train


# Helper function to create test data
def _create_test_data():
    """Create test data for smart_train tests."""
    torch.manual_seed(42)
    X = torch.randn(16, 4).to(device)
    y = torch.randint(0, 2, (16,)).to(device)
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=8)
    model = nn.Linear(4, 2).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()
    return model, optimizer, loss_fn, dataloader


def test_smart_train_no_features():
    """Test smart_train with no additional features."""
    model, optimizer, loss_fn, dataloader = _create_test_data()
    
    result = smart_train(model, optimizer, loss_fn, dataloader)
    
    assert isinstance(result, dict)
    assert 'model' in result
    assert isinstance(result['model'], nn.Module)


def test_smart_train_single_feature():
    """Test smart_train with single feature (direct execution)."""
    model, optimizer, loss_fn, dataloader = _create_test_data()
    
    result = smart_train(
        model, optimizer, loss_fn, dataloader,
        accum_steps=2
    )
    
    assert isinstance(result, dict)
    assert 'model' in result
    assert isinstance(result['model'], nn.Module)


def test_smart_train_multi_feature():
    """Test smart_train with multiple features (compilation or demo mode)."""
    model, optimizer, loss_fn, dataloader = _create_test_data()
    
    result = smart_train(
        model, optimizer, loss_fn, dataloader,
        accum_steps=2, track_loss=True
    )
    
    assert isinstance(result, dict)
    assert 'model' in result
    assert isinstance(result['model'], nn.Module)


def test_smart_train_unknown_kwargs():
    """Test that smart_train rejects unknown kwargs."""
    model, optimizer, loss_fn, dataloader = _create_test_data()
    
    with pytest.raises(ValueError) as exc_info:
        smart_train(
            model, optimizer, loss_fn, dataloader,
            unknown_parameter=123
        )
    
    error_msg = str(exc_info.value)
    assert "Unknown kwargs" in error_msg


def test_smart_train_none_filtering():
    """Test that smart_train filters out None values."""
    model, optimizer, loss_fn, dataloader = _create_test_data()
    
    # Should work the same as no additional features
    result = smart_train(
        model, optimizer, loss_fn, dataloader,
        accum_steps=None, val_loader=None
    )
    
    assert isinstance(result, dict)
    assert 'model' in result