"""
Comprehensive tests for smart_api.py functionality.
Tests the visual validations as proper unit tests.
"""

from typing import Dict, Any
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pytest
from unittest.mock import MagicMock
from importlib import reload

from gpt_lab.train_loops.tests.test_utils import SimpleTestTrainingModel, AVAILABLE_DEVICES


# Helper function to create test data
def _create_test_data(device: str):
    """Create test data for smart_train tests."""
    torch.manual_seed(42)
    X = torch.randn(16, 4).to(device)
    y = torch.randint(0, 2, (16,)).to(device)
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=8)
    backbone = nn.Linear(4, 2)
    loss_fn = nn.CrossEntropyLoss()
    model = SimpleTestTrainingModel(backbone, loss_fn).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    return model, optimizer, dataloader


@pytest.mark.parametrize("device", AVAILABLE_DEVICES)
def test_smart_train_no_features(device: str):
    """Test smart_train with no additional features."""
    from gpt_lab.train_loops.smart_api import smart_train
    
    model, optimizer, dataloader = _create_test_data(device)
    
    result = smart_train(model, optimizer, dataloader)
    
    assert isinstance(result, dict)
    assert 'model' in result
    assert isinstance(result['model'], nn.Module)


@pytest.mark.parametrize("device", AVAILABLE_DEVICES)
def test_smart_train_single_feature(device: str):
    """Test smart_train with single feature (direct execution)."""
    from gpt_lab.train_loops.smart_api import smart_train
    
    model, optimizer, dataloader = _create_test_data(device)
    
    result = smart_train(
        model, optimizer, dataloader,
        accum_steps=2
    )
    
    assert isinstance(result, dict)
    assert 'model' in result
    assert isinstance(result['model'], nn.Module)


@pytest.mark.parametrize("device", AVAILABLE_DEVICES)
def test_smart_train_multi_feature(device: str):
    """Test smart_train with multiple features (uses cached or real compilation)."""
    from gpt_lab.train_loops.smart_api import smart_train
    
    model, optimizer, dataloader = _create_test_data(device)
    
    # This test uses the real compilation chain (or cached results)
    # We're just verifying that smart_train works with multiple features
    result = smart_train(
        model, optimizer, dataloader,
        accum_steps=2, track_loss=True
    )
    
    # Assert that training completed and returned a valid result
    assert isinstance(result, dict)
    assert 'model' in result
    assert isinstance(result['model'], nn.Module)


def test_smart_train_unknown_kwargs():
    """Test that smart_train rejects unknown kwargs."""
    from gpt_lab.train_loops.smart_api import smart_train
    
    # This test is device-agnostic, so we can just use CPU
    model, optimizer, dataloader = _create_test_data('cpu')
    
    with pytest.raises(ValueError) as exc_info:
        smart_train(
            model, optimizer, dataloader,
            unknown_parameter=123
        )
    
    error_msg = str(exc_info.value)
    assert "Unknown kwargs" in error_msg


@pytest.mark.parametrize("device", AVAILABLE_DEVICES)
def test_smart_train_none_filtering(device: str):
    """Test that smart_train filters out None values."""
    from gpt_lab.train_loops.smart_api import smart_train
    
    model, optimizer, dataloader = _create_test_data(device)
    
    # Should work the same as no additional features
    result = smart_train(
        model, optimizer, dataloader,
        accum_steps=None, val_loader=None
    )
    
    assert isinstance(result, dict)
    assert 'model' in result