from typing import Dict, Any, List, Callable
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pytest

from utils.device import get_available_devices
from utils.testing import list_all_files_in_folder_and_subdirs, import_module_from_path


parent_dir = os.path.dirname(os.path.abspath(__file__))
catalog_dir = os.path.join(parent_dir, "catalog")
all_loop_files = list_all_files_in_folder_and_subdirs(catalog_dir)
all_loop_files = [loop_file for loop_file in all_loop_files 
                    if (loop_file[-11:] != "__init__.py" and loop_file[-8:] != "_test.py")]

# Import modules and extract run_training functions
all_loop_modules = []
all_loop_functions = []
for loop_file in all_loop_files:
    module = import_module_from_path(f"loop_module_{len(all_loop_modules)}", os.path.join(catalog_dir, loop_file))
    if hasattr(module, 'run_training'):
        all_loop_modules.append(module)
        all_loop_functions.append(module.run_training)


def discover_specific_tests() -> Dict[str, List[Callable]]:
    """Discover all specific test functions for atomic features."""
    specific_tests = {}
    atomic_features_dir = Path("src/train_loops/catalog/atomic_features")
    
    for test_file in atomic_features_dir.glob("*_test.py"):
        feature_name = test_file.stem.replace("_test", "")
        try:
            module = import_module_from_path(f"test_{feature_name}", test_file)
            if hasattr(module, "__specific_tests__"):
                specific_tests[feature_name] = module.__specific_tests__
        except Exception as e:
            print(f"Warning: Failed to load specific tests from {test_file}: {e}")
    
    return specific_tests


def discover_atomic_features() -> List[Callable]:
    """Discover all atomic feature run_training functions."""
    atomic_features_dir = Path("src/train_loops/catalog/atomic_features")
    atomic_functions = []
    
    for feature_file in atomic_features_dir.glob("*.py"):
        # Skip test files, __init__.py, and base_loop.py
        if (feature_file.name.endswith("_test.py") or 
            feature_file.name == "__init__.py" or
            feature_file.name == "base_loop.py"):
            continue
            
        try:
            module = import_module_from_path(f"atomic_{feature_file.stem}", feature_file)
            if hasattr(module, 'run_training'):
                atomic_functions.append((module.run_training, feature_file.stem))
        except Exception as e:
            print(f"Warning: Failed to load atomic feature from {feature_file}: {e}")
    
    return atomic_functions


def generate_compiled_loop_specific_tests():
    """Generate pytest parameters for specific tests on compiled loops."""
    specific_tests = discover_specific_tests()
    compiled_dir = Path("src/train_loops/catalog/llm_compiled")
    params = []
    AVAILABLE_DEVICES, _ = get_available_devices()
    
    for compiled_loop_file in compiled_dir.glob("*.py"):
        try:
            # Load the module to get atomic features
            module = import_module_from_path(f"compiled_test_{compiled_loop_file.stem}", compiled_loop_file)
            atomic_features = getattr(module, '__atomic_features__', [])
            
            # For each atomic feature, add its specific tests for each device
            for feature in atomic_features:
                if feature in specific_tests:
                    for test_func in specific_tests[feature]:
                        for device in AVAILABLE_DEVICES:
                            params.append(pytest.param(
                                test_func, module.run_training, str(compiled_loop_file), feature, device,
                                id=f"{compiled_loop_file.stem}_{feature}_{test_func.__name__}_{device}"
                            ))
        except Exception as e:
            print(f"Warning: Failed to process compiled loop {compiled_loop_file}: {e}")
    
    return params


def base_loop_compliance_test(run_training_fn, feature_name: str, device: str):
    """
    Test that an atomic feature with default arguments behaves identically to base_loop.py.
    This ensures all atomic features maintain backward compatibility and follow the standard.
    """
    # Import base_loop for comparison
    base_loop_path = Path("src/train_loops/catalog/atomic_features/base_loop.py")
    base_module = import_module_from_path("base_loop_ref", base_loop_path)
    base_run_training = base_module.run_training
    
    # Create deterministic test setup
    torch.manual_seed(42)
    X = torch.randn(128, 16).to(device)
    y = torch.randint(0, 3, (128,)).to(device)
    ds = TensorDataset(X, y)
    dl = DataLoader(ds, batch_size=32, shuffle=False)  # No shuffle for deterministic behavior
    
    # Test with two identical models to compare behaviors
    torch.manual_seed(42)
    model1 = nn.Sequential(nn.Linear(16, 8), nn.ReLU(), nn.Linear(8, 3))
    model1.to(device)
    optimizer1 = torch.optim.SGD(model1.parameters(), lr=0.01)
    loss_fn1 = nn.CrossEntropyLoss()
    
    torch.manual_seed(42)  # Reset seed for identical initialization
    model2 = nn.Sequential(nn.Linear(16, 8), nn.ReLU(), nn.Linear(8, 3))
    model2.to(device)
    optimizer2 = torch.optim.SGD(model2.parameters(), lr=0.01)
    loss_fn2 = nn.CrossEntropyLoss()
    
    # Run base_loop
    base_result = base_run_training(
        model=model1,
        optimizer=optimizer1,
        loss_fn=loss_fn1,
        train_loader=dl,
    )
    
    # Run atomic feature with default arguments (no extra kwargs)
    feature_result = run_training_fn(
        model=model2,
        optimizer=optimizer2,
        loss_fn=loss_fn2,
        train_loader=dl,
    )
    
    # Verify both results are dicts with 'model' key
    if not isinstance(base_result, dict) or 'model' not in base_result:
        raise AssertionError("base_loop.py must return dict with 'model' key")
    if not isinstance(feature_result, dict) or 'model' not in feature_result:
        raise AssertionError(f"{feature_name} must return dict with 'model' key")
    
    # Compare model parameters to ensure identical training occurred
    base_params = list(base_result['model'].parameters())
    feature_params = list(feature_result['model'].parameters())
    
    if len(base_params) != len(feature_params):
        raise AssertionError(f"{feature_name}: Model parameter count mismatch with base_loop")
    
    for i, (base_p, feature_p) in enumerate(zip(base_params, feature_params)):
        if not torch.allclose(base_p.data, feature_p.data, atol=1e-6, rtol=1e-5):
            raise AssertionError(
                f"{feature_name}: Parameter {i} differs from base_loop behavior. "
                f"This indicates the default arguments don't produce base_loop-equivalent behavior. "
                f"Max diff: {torch.max(torch.abs(base_p.data - feature_p.data)).item():.2e}"
            )
    
    # For atomic features, the result dict should only contain 'model' when using defaults
    # (unless the feature inherently changes the return format, like mixed_precision with used_amp)
    expected_keys = {'model'}
    
    # Allow certain features to have additional keys even with defaults disabled
    if feature_name == 'mixed_precision':
        expected_keys.add('used_amp')  # This key is always present to indicate AMP status
    
    extra_keys = set(feature_result.keys()) - expected_keys
    if extra_keys:
        raise AssertionError(
            f"{feature_name}: With default arguments, should only return {expected_keys} keys, "
            f"but also returned: {extra_keys}. This suggests default arguments don't disable the feature."
        )


def universal_learning_test(run_training_fn, device: str):
    """
    Build a tiny task and ensure real learning happened (loss drops).
    This is a standalone function for use by the LLM compiler.
    """
    torch.manual_seed(0)
    X = torch.randn(2048, 32).to(device)
    y = (X.sum(dim=1) > 0).long().to(device)
    ds = TensorDataset(X, y)
    dl = DataLoader(ds, batch_size=64, shuffle=True)

    model = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 2))
    loss_fn = nn.CrossEntropyLoss()
    optim = torch.optim.AdamW(model.parameters(), lr=3e-3)

    model.to(device)
    with torch.no_grad():
        pre = loss_fn(model(X.to(device)), y.to(device)).item()

    result = run_training_fn(
        model=model,
        optimizer=optim,
        loss_fn=loss_fn,
        train_loader=dl,
    )

    with torch.no_grad():
        post = loss_fn(model(X.to(device)), y.to(device)).item()

    if not isinstance(result, dict):
        raise AssertionError("run_training(...) must return dict metrics.")
    if 'model' not in result:
        raise AssertionError("The result dictionary must contain the key 'model'.")
    if not isinstance(result['model'], nn.Module):
        raise AssertionError("The 'model' key in the result dictionary must be an instance of nn.Module.")
    if not (post < pre * 0.9):  # at least 10% relative improvement
        raise AssertionError(f"Training did not sufficiently improve loss: pre={pre:.4f}, post={post:.4f}")


# Get available devices for parameterization
AVAILABLE_DEVICES, _ = get_available_devices()

# Create parameterized tests for universal learning across devices
universal_test_params = []
for run_training_fn in all_loop_functions:
    for device in AVAILABLE_DEVICES:
        universal_test_params.append(
            pytest.param(run_training_fn, device, id=f"{run_training_fn.__module__}_{device}")
        )

# Create parameterized tests for atomic feature compliance across devices
atomic_compliance_params = []
for fn, name in discover_atomic_features():
    for device in AVAILABLE_DEVICES:
        atomic_compliance_params.append(
            pytest.param(fn, name, device, id=f"{name}_{device}")
        )


@pytest.mark.parametrize("run_training_fn,device", universal_test_params)
def test_universal_learning_pytest(run_training_fn, device):
    """
    Pytest wrapper for the universal learning test.
    """
    universal_learning_test(run_training_fn, device)


# Test that all atomic features behave like base_loop.py with default arguments
@pytest.mark.parametrize("run_training_fn,feature_name,device", atomic_compliance_params)
def test_atomic_feature_base_compliance(run_training_fn, feature_name, device):
    """
    Test that atomic features with default arguments behave identically to base_loop.py.
    This enforces the standard that all atomic features must be backwards compatible.
    """
    base_loop_compliance_test(run_training_fn, feature_name, device)


# Add new parameterized test for compiled loops
@pytest.mark.parametrize("test_func,run_training_fn,loop_file,source_feature,device", 
                        generate_compiled_loop_specific_tests())
def test_compiled_loop_specific_behaviors(test_func, run_training_fn, loop_file, source_feature, device):
    """Run specific tests from atomic features on compiled loops that use them."""
    test_func(run_training_fn, device)