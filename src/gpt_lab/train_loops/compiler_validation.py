from typing import Dict, List, Callable, Any
from pathlib import Path
import inspect
import importlib
import pkgutil

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, IterableDataset

from gpt_lab.catalog_utils import import_module_from_path
from gpt_lab.catalog_bootstrap import get_active_context
from gpt_lab.device import get_available_devices


class SimpleTestTrainingModel(nn.Module):
    """A simple nn.Module wrapper for testing that acts as a TrainingModel."""
    def __init__(self, backbone, loss_fn):
        super().__init__()
        self.backbone = backbone
        self.loss_fn = loss_fn

    def forward(self, batch):
        xb, yb = batch
        logits = self.backbone(xb)
        return self.loss_fn(logits, yb)

# Discover available devices once and export for all tests to use
AVAILABLE_DEVICES, _ = get_available_devices()


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


def discover_specific_tests() -> Dict[str, List[Callable]]:
    """Discover all specific test functions for atomic features."""
    specific_tests: Dict[str, List[Callable]] = {}
    active_context = get_active_context()

    # Discover tests colocated next to or within a tests/ subdir of atomic features
    for root in active_context.get("ordered_roots", []):
        train_loops_dir = Path(root) / "train_loops"
        if not train_loops_dir.is_dir():
            continue

        # Define all paths to search for tests
        search_paths = [train_loops_dir]
        tests_subdir = train_loops_dir / "tests"
        if tests_subdir.is_dir():
            search_paths.append(tests_subdir)

        for search_path in search_paths:
            test_files = list(search_path.glob("test_*.py")) + list(search_path.glob("*_test.py"))

            for test_file in test_files:
                if test_file.stem.startswith("test_"):
                    feature_name = test_file.stem[5:]
                else:
                    feature_name = test_file.stem.replace("_test", "")

                try:
                    # Create a unique module name to avoid collisions
                    module_name = f"test_{feature_name}_{root.name}_{search_path.name}"
                    module = import_module_from_path(module_name, test_file)
                    if hasattr(module, "__specific_tests__"):
                        if feature_name not in specific_tests:
                            specific_tests[feature_name] = []
                        # Use extend to accumulate tests from all locations, enforcing the contract
                        specific_tests[feature_name].extend(module.__specific_tests__)
                except Exception:
                    # Silently skip modules that fail to import
                    pass

    return specific_tests


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
