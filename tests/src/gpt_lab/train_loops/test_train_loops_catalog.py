from typing import Dict, List, Callable
from pathlib import Path
import inspect
import importlib
import pkgutil

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, IterableDataset
import pytest

from gpt_lab.catalog_utils import import_module_from_path
from gpt_lab.catalog_bootstrap import get_all_artifact_roots_for_active
from gpt_lab.train_loops.compiler_validation import (
    universal_learning_test,
    discover_specific_tests,
    base_loop_compliance_test,
    dataset_type_compatibility_test,
    AVAILABLE_DEVICES
)


def _iter_train_loop_modules():
    try:
        pkg = importlib.import_module("gpt_lab.train_loops")
    except Exception:
        return []
    mods = []
    for _, name, _ in pkgutil.walk_packages(pkg.__path__, prefix=pkg.__name__ + "."):
        # Skip tests packages
        if ".tests" in name:
            continue
        try:
            m = importlib.import_module(name)
            if hasattr(m, 'run_training'):
                mods.append(m)
        except Exception:
            continue
    return mods

all_loop_modules = _iter_train_loop_modules()
all_loop_functions = [m.run_training for m in all_loop_modules]


def discover_atomic_features() -> List[Callable]:
    """Discover all atomic feature run_training functions."""
    atomic_functions = []
    try:
        train_pkg = importlib.import_module("gpt_lab.train_loops")
    except Exception:
        return []
    for _, name, _ in pkgutil.walk_packages(train_pkg.__path__, prefix=train_pkg.__name__ + "."):
        leaf = name.split(".")[-1]
        if leaf.endswith("_test") or leaf in ("__init__", "base_loop"):
            continue
        try:
            m = importlib.import_module(name)
            if hasattr(m, 'run_training'):
                atomic_functions.append((m.run_training, leaf))
        except Exception as e:
            print(f"Warning: Failed to load atomic feature from {name}: {e}")
    
    return atomic_functions


def discover_compiled_loops() -> List[tuple]:
    """
    Discover all compiled training loops from artifact directories.
    Returns list of (run_training_fn, loop_name, loop_path) tuples.
    """
    compiled_loops = []
    compiled_files = []
    
    # Search artifacts across all active roots
    for art_root in get_all_artifact_roots_for_active():
        cand = art_root / "train_loops" / "llm_compiled"
        if cand.is_dir():
            compiled_files.extend(sorted(cand.glob("*.py")))
    
    for compiled_path in compiled_files:
        try:
            module = import_module_from_path(f"compiled_loop_{compiled_path.stem}", compiled_path)
            if hasattr(module, 'run_training'):
                # Use the filename as the loop name
                loop_name = f"compiled_{compiled_path.stem}"
                compiled_loops.append((module.run_training, loop_name, str(compiled_path)))
        except Exception as e:
            print(f"Warning: Failed to load compiled loop from {compiled_path}: {e}")
    
    return compiled_loops


def generate_compiled_loop_specific_tests():
    """Generate pytest parameters for specific tests on compiled loops."""
    specific_tests = discover_specific_tests()
    params = []
    # Search artifacts across all active roots
    compiled_files = []
    for art_root in get_all_artifact_roots_for_active():
        cand = art_root / "train_loops" / "llm_compiled"
        if cand.is_dir():
            compiled_files.extend(sorted(cand.glob("*.py")))
    for compiled_path in compiled_files:
        try:
            module = import_module_from_path(f"compiled_test_{compiled_path.stem}", compiled_path)
            atomic_features = getattr(module, '__atomic_features__', [])
            
            for feature in atomic_features:
                if feature in specific_tests:
                    for test_func in specific_tests[feature]:
                        # Check if test function has compatible signature
                        # It should accept exactly (run_training_fn, device) - no pytest fixtures
                        sig = inspect.signature(test_func)
                        params_list = list(sig.parameters.keys())
                        
                        # Skip tests that require pytest fixtures (more than 2 params, or params like tmp_path, monkeypatch, etc.)
                        if len(params_list) != 2:
                            continue
                        if any(p in params_list for p in ['tmp_path', 'monkeypatch', 'request', 'capsys', 'capfd']):
                            continue
                        
                        for device in AVAILABLE_DEVICES:
                            params.append(pytest.param(
                                test_func, module.run_training, str(compiled_path), feature, device,
                                id=f"{compiled_path.stem}_{feature}_{test_func.__name__}_{device}"
                            ))
        except Exception as e:
            print(f"Warning: Failed to process compiled loop at {compiled_path}: {e}")
    return params


# Create parameterized tests for universal learning across devices
universal_test_params = []
for run_training_fn in all_loop_functions:
    for device in AVAILABLE_DEVICES:
        universal_test_params.append(
            pytest.param(run_training_fn, device, id=f"{run_training_fn.__module__}_{device}")
        )

# Add compiled loops to universal tests
for fn, name, path in discover_compiled_loops():
    for device in AVAILABLE_DEVICES:
        universal_test_params.append(
            pytest.param(fn, device, id=f"{name}_{device}")
        )


# Create parameterized tests for atomic feature compliance across devices
atomic_compliance_params = []
for fn, name in discover_atomic_features():
    for device in AVAILABLE_DEVICES:
        atomic_compliance_params.append(
            pytest.param(fn, name, device, id=f"{name}_{device}")
        )

# Add compiled loops to base compliance tests
for fn, name, path in discover_compiled_loops():
    for device in AVAILABLE_DEVICES:
        atomic_compliance_params.append(
            pytest.param(fn, name, device, id=f"{name}_compliance_{device}")
        )


# Create parameterized tests for dataset compatibility across devices
dataset_type_compatibility_params = []
for run_training_fn in all_loop_functions:
    for device in AVAILABLE_DEVICES:
        dataset_type_compatibility_params.append(
            pytest.param(run_training_fn, device, id=f"{run_training_fn.__module__}_dataset_compat_{device}")
        )

# Add compiled loops to dataset compatibility tests
for fn, name, path in discover_compiled_loops():
    for device in AVAILABLE_DEVICES:
        dataset_type_compatibility_params.append(
            pytest.param(fn, device, id=f"{name}_dataset_compat_{device}")
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


@pytest.mark.parametrize("run_training_fn,device", dataset_type_compatibility_params)
def test_dataset_type_compatibility_pytest(run_training_fn, device):
    """Pytest wrapper for the dataset compatibility test."""
    dataset_type_compatibility_test(run_training_fn, device)


# Add new parameterized test for compiled loops
@pytest.mark.parametrize("test_func,run_training_fn,loop_file,source_feature,device", 
                        generate_compiled_loop_specific_tests())
def test_compiled_loop_specific_behaviors(test_func, run_training_fn, loop_file, source_feature, device):
    """Run specific tests from atomic features on compiled loops that use them."""
    test_func(run_training_fn, device)