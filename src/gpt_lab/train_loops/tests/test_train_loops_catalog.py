from typing import Dict, Any, List, Callable
import os
import sys
from pathlib import Path
import inspect

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, IterableDataset
import pytest

from gpt_lab.catalog_utils import list_all_files_in_folder_and_subdirs, import_module_from_path
import importlib
import pkgutil
from gpt_lab.catalog_bootstrap import get_all_artifact_roots_for_active
from gpt_lab.train_loops.tests.test_utils import SimpleTestTrainingModel, AVAILABLE_DEVICES


# --- Path Constants ---
TESTS_ROOT = Path(__file__).parent


class SimpleIterableDataset(IterableDataset):
    """A simple iterable dataset for testing purposes."""
    def __init__(self, X, y, batch_size):
        super().__init__()
        assert X.shape[0] == y.shape[0]
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.num_samples = X.shape[0]

    def __iter__(self):
        for i in range(0, self.num_samples, self.batch_size):
            end = min(i + self.batch_size, self.num_samples)
            yield self.X[i:end], self.y[i:end]


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


def discover_specific_tests() -> Dict[str, List[Callable]]:
    """Discover all specific test functions for atomic features."""
    specific_tests: Dict[str, List[Callable]] = {}
    
    # 1) Discover tests colocated in repo tests folder (back-compat)
    atomic_tests_dir = TESTS_ROOT / "catalog" / "atomic_features"
    if atomic_tests_dir.exists():
        # Support both test_*.py (prefix) and *_test.py (suffix) naming patterns
        for test_file in list(atomic_tests_dir.glob("test_*.py")) + list(atomic_tests_dir.glob("*_test.py")):
            # Extract feature name from both patterns
            if test_file.stem.startswith("test_"):
                feature_name = test_file.stem[5:]  # Remove "test_" prefix
            else:
                feature_name = test_file.stem.replace("_test", "")  # Remove "_test" suffix
            try:
                module = import_module_from_path(f"test_{feature_name}", test_file)
                if hasattr(module, "__specific_tests__"):
                    specific_tests[feature_name] = module.__specific_tests__
            except Exception as e:
                print(f"Warning: Failed to load specific tests from {test_file}: {e}")

    # 2) Discover tests colocated next to atomic features across all active roots
    try:
        train_pkg = importlib.import_module("gpt_lab.train_loops")
        for _, name, _ in pkgutil.walk_packages(train_pkg.__path__, prefix=train_pkg.__name__ + "."):
            leaf = name.split(".")[-1]
            # Support both test_* (prefix) and *_test (suffix) naming patterns
            if not (leaf.endswith("_test") or leaf.startswith("test_")):
                continue
            try:
                m = importlib.import_module(name)
                if hasattr(m, "__specific_tests__"):
                    # Extract feature name from both patterns
                    if leaf.startswith("test_"):
                        feature_name = leaf[5:]  # Remove "test_" prefix
                    else:
                        feature_name = leaf.replace("_test", "")  # Remove "_test" suffix
                    specific_tests[feature_name] = m.__specific_tests__
            except Exception as e:
                print(f"Warning: Failed to load specific tests from module {name}: {e}")
    except Exception:
        pass

    return specific_tests


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


def base_loop_compliance_test(run_training_fn, feature_name: str, device: str):
    """
    Test that an atomic feature with default arguments behaves identically to base_loop.py.
    This ensures all atomic features maintain backward compatibility and follow the standard.
    """
    # Import base_loop for comparison via namespace
    base_module = importlib.import_module("gpt_lab.train_loops.base_loop")
    base_run_training = getattr(base_module, 'run_training')
    
    # Create deterministic test setup
    torch.manual_seed(42)
    X = torch.randn(128, 16).to(device)
    y = torch.randint(0, 3, (128,)).to(device)
    ds = TensorDataset(X, y)
    dl = DataLoader(ds, batch_size=32, shuffle=False)  # No shuffle for deterministic behavior
    
    # Test with two identical models to compare behaviors
    torch.manual_seed(42)
    backbone1 = nn.Sequential(nn.Linear(16, 8), nn.ReLU(), nn.Linear(8, 3))
    model1 = SimpleTestTrainingModel(backbone1, nn.CrossEntropyLoss()).to(device)
    optimizer1 = torch.optim.SGD(model1.parameters(), lr=0.01)
    
    torch.manual_seed(42)  # Reset seed for identical initialization
    backbone2 = nn.Sequential(nn.Linear(16, 8), nn.ReLU(), nn.Linear(8, 3))
    model2 = SimpleTestTrainingModel(backbone2, nn.CrossEntropyLoss()).to(device)
    optimizer2 = torch.optim.SGD(model2.parameters(), lr=0.01)
    
    # Run base_loop
    base_result = base_run_training(
        model=model1,
        optimizer=optimizer1,
        train_loader=dl,
    )
    
    # Run atomic feature with default arguments (no extra kwargs)
    feature_result = run_training_fn(
        model=model2,
        optimizer=optimizer2,
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

    backbone = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 2))
    model = SimpleTestTrainingModel(backbone, nn.CrossEntropyLoss()).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=3e-3)

    model.to(device)
    with torch.no_grad():
        pre = model((X.to(device), y.to(device))).item()

    result = run_training_fn(
        model=model,
        optimizer=optim,
        train_loader=dl,
    )

    with torch.no_grad():
        post = model((X.to(device), y.to(device))).item()

    if not isinstance(result, dict):
        raise AssertionError("run_training(...) must return dict metrics.")
    if 'model' not in result:
        raise AssertionError("The result dictionary must contain the key 'model'.")
    if not isinstance(result['model'], nn.Module):
        raise AssertionError("The 'model' key in the result dictionary must be an instance of nn.Module.")
    if not (post < pre * 0.9):  # at least 10% relative improvement
        raise AssertionError(f"Training did not sufficiently improve loss: pre={pre:.4f}, post={post:.4f}")


def dataset_type_compatibility_test(run_training_fn, device: str):
    """
    Tests that a training loop works with both map-style (indexed) and iterable-style datasets
    by running a consistent learning task on both.
    """
    def run_test_on_loader(loader, X_full, y_full, model, optim, ds_type: str):
        """Helper to run the learning test on a given dataloader."""
        model.to(device)
        with torch.no_grad():
            pre_loss = model((X_full, y_full)).item()
        
        try:
            run_training_fn(
                model=model,
                optimizer=optim,
                train_loader=loader,
            )
        except Exception as e:
            raise AssertionError(f"Training loop failed with a {ds_type} dataset: {e}") from e

        with torch.no_grad():
            post_loss = model((X_full, y_full)).item()

        if not (post_loss < pre_loss * 0.9):  # at least 10% relative improvement
            raise AssertionError(
                f"Loss did not sufficiently decrease for {ds_type} dataset. "
                f"pre={pre_loss:.4f}, post={post_loss:.4f}"
            )

    batch_size = 64
    torch.manual_seed(0)
    X = torch.randn(2048, 32).to(device)
    y = (X.sum(dim=1) > 0).long().to(device)

    # --- 1. Test with Map-style Dataset (TensorDataset) ---
    torch.manual_seed(1)
    backbone_map = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 2))
    model_map = SimpleTestTrainingModel(backbone_map, nn.CrossEntropyLoss())
    optim_map = torch.optim.AdamW(model_map.parameters(), lr=3e-3)
    
    map_dataset = TensorDataset(X, y)
    map_loader = DataLoader(map_dataset, batch_size=batch_size, shuffle=True)
    run_test_on_loader(map_loader, X, y, model_map, optim_map, "map-style")

    # --- 2. Test with Iterable-style Dataset ---
    torch.manual_seed(1) # Reset seed for identical model initialization
    backbone_iter = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 2))
    model_iter = SimpleTestTrainingModel(backbone_iter, nn.CrossEntropyLoss())
    optim_iter = torch.optim.AdamW(model_iter.parameters(), lr=3e-3)
    
    iterable_dataset = SimpleIterableDataset(X, y, batch_size=batch_size)
    iterable_loader = DataLoader(iterable_dataset, batch_size=None) # batch_size=None as dataset yields batches
    run_test_on_loader(iterable_loader, X, y, model_iter, optim_iter, "iterable-style")


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