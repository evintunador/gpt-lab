from typing import Dict, Any, List, Callable
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pytest

from utils.device import best_device as device
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
    atomic_features_dir = Path("train_loops/catalog/atomic_features")
    
    for test_file in atomic_features_dir.glob("*_test.py"):
        feature_name = test_file.stem.replace("_test", "")
        try:
            module = import_module_from_path(f"test_{feature_name}", test_file)
            if hasattr(module, "__specific_tests__"):
                specific_tests[feature_name] = module.__specific_tests__
        except Exception as e:
            print(f"Warning: Failed to load specific tests from {test_file}: {e}")
    
    return specific_tests


def generate_compiled_loop_specific_tests():
    """Generate pytest parameters for specific tests on compiled loops."""
    specific_tests = discover_specific_tests()
    compiled_dir = Path("train_loops/catalog/llm_compiled")
    params = []
    
    for compiled_loop_file in compiled_dir.glob("*.py"):
        try:
            # Load the module to get atomic features
            module = import_module_from_path(f"compiled_test_{compiled_loop_file.stem}", compiled_loop_file)
            atomic_features = getattr(module, '__atomic_features__', [])
            
            # For each atomic feature, add its specific tests
            for feature in atomic_features:
                if feature in specific_tests:
                    for test_func in specific_tests[feature]:
                        params.append(pytest.param(
                            test_func, module.run_training, str(compiled_loop_file), feature,
                            id=f"{compiled_loop_file.stem}_{feature}_{test_func.__name__}"
                        ))
        except Exception as e:
            print(f"Warning: Failed to process compiled loop {compiled_loop_file}: {e}")
    
    return params


@pytest.mark.parametrize("run_training_fn", all_loop_functions)
def test_universal_learning(run_training_fn):
    """
    Build a tiny task and ensure real learning happened (loss drops).
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


# Add new parameterized test for compiled loops
@pytest.mark.parametrize("test_func,run_training_fn,loop_file,source_feature", 
                        generate_compiled_loop_specific_tests())
def test_compiled_loop_specific_behaviors(test_func, run_training_fn, loop_file, source_feature):
    """Run specific tests from atomic features on compiled loops that use them."""
    test_func(run_training_fn, device)