from typing import Dict, Any
import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pytest

from utils.device import best_device as device
from utils.testing import list_all_files_in_folder_and_subdirs, import_module_from_path


parent_dir = os.path.dirname(os.path.abspath(__file__))
catalog_dir = os.path.join(parent_dir, "catalog")
all_loop_files = list_all_files_in_folder_and_subdirs(catalog_dir)
all_loop_files = [loop_file for loop_file in all_loop_files if loop_file[-11:] != "__init__.py"]

# Import modules and extract run_training functions
all_loop_modules = []
all_loop_functions = []
for loop_file in all_loop_files:
    module = import_module_from_path(f"loop_module_{len(all_loop_modules)}", os.path.join(catalog_dir, loop_file))
    if hasattr(module, 'run_training'):
        all_loop_modules.append(module)
        all_loop_functions.append(module.run_training)


def universal_learning_test(run_training_fn, device=device) -> Dict[str, Any]:
    """
    Build a tiny task and ensure real learning happened (loss drops).
    Returns metrics dict for use in other contexts (like LLM compiler).
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
    # Measure pre-training loss
    with torch.no_grad():
        pre = loss_fn(model(X.to(device)), y.to(device)).item()

    result = run_training_fn(
        model=model,
        optimizer=optim,
        loss_fn=loss_fn,
        train_loader=dl,
    )

    # Measure post-training loss
    with torch.no_grad():
        post = loss_fn(model(X.to(device)), y.to(device)).item()

    if not isinstance(result, dict):
        raise AssertionError("run_training(...) must return dict metrics.")
    if not (post < pre * 0.9):  # at least 10% relative improvement
        raise AssertionError(f"Training did not sufficiently improve loss: pre={pre:.4f}, post={post:.4f}")

    return {"pre_loss": pre, "post_loss": post, **result}


@pytest.mark.parametrize("run_training_fn,loop_file", zip(all_loop_functions, all_loop_files[:len(all_loop_functions)]))
def test_universal_learning(run_training_fn, loop_file):
    """
    Pytest wrapper that calls universal_learning_test but returns None.
    """
    # Call the test function but don't return its result
    universal_learning_test(run_training_fn)
    # Pytest test functions should return None