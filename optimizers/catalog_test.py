from typing import Dict, Any, List, Tuple, Type, Optional
import os
import sys
import inspect

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pytest

from device import get_default_device
from catalog_utils import list_all_files_in_folder_and_subdirs, import_module_from_path
from optimizers.catalog_utils import OptimizerConfig, create_smart_optimizer


device = get_default_device()


parent_dir = os.path.dirname(os.path.abspath(__file__))
catalog_dir = os.path.join(parent_dir, "catalog")
all_optimizer_files = list_all_files_in_folder_and_subdirs(catalog_dir)
all_optimizer_files = [opt_file for opt_file in all_optimizer_files if opt_file != "__init__.py"]

# Import modules and extract optimizer classes
all_optimizer_modules = []
all_optimizer_classes = []
all_optimizer_configs = []

for optimizer_file in all_optimizer_files:
    try:
        module = import_module_from_path(f"optimizer_module_{len(all_optimizer_modules)}", os.path.join(catalog_dir, optimizer_file))
        all_optimizer_modules.append(module)
        
        # Look for optimizer classes in the module
        for name, obj in inspect.getmembers(module):
            if (inspect.isclass(obj) and 
                issubclass(obj, torch.optim.Optimizer) and 
                obj != torch.optim.Optimizer):
                
                # Get default config if available, otherwise None for auto-detection
                test_config = getattr(module, '__test_config__', None)
                
                all_optimizer_classes.append(obj)
                all_optimizer_configs.append(test_config)
                break
    except Exception as e:
        print(f"[WARNING] Failed to import optimizer from {optimizer_file}: {e}")


def test_optimizer_discovery():
    """Test that we successfully discovered optimizers from the catalog."""
    if not all_optimizer_classes:
        pytest.fail(
            f"No optimizer classes discovered from {len(all_optimizer_files)} files in catalog/. "
            f"Files found: {all_optimizer_files}"
        )


def universal_optimizer_test(
    optimizer_class: Type[torch.optim.Optimizer], 
    config: Optional[OptimizerConfig],
    device: str = device
) -> Dict[str, Any]:
    """
    Build a tiny task and ensure the optimizer can learn it (loss drops).
    Uses smart optimizer creation that handles constraints gracefully.
    """
    torch.manual_seed(42)
    
    # Create a simple but non-trivial learning task
    X = torch.randn(2048, 32).to(device)
    true_weights = torch.randn(32).to(device)
    threshold = 0.0
    y = ((X * true_weights).sum(dim=1) > threshold).long().to(device)
    
    ds = TensorDataset(X, y)
    dl = DataLoader(ds, batch_size=64, shuffle=True)
    
    model = nn.Sequential(
        nn.Linear(32, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 2)
    ).to(device)
    
    loss_fn = nn.CrossEntropyLoss()
    
    def measure_loss(model, dataloader):
        model.eval()
        with torch.no_grad():
            total_loss = 0.0
            total_samples = 0
            for batch_X, batch_y in dataloader:
                outputs = model(batch_X)
                loss = loss_fn(outputs, batch_y)
                total_loss += loss.item() * batch_X.size(0)
                total_samples += batch_X.size(0)
        return total_loss / total_samples
    
    # Measure initial loss
    pre_loss = measure_loss(model, dl)
    
    # Test the optimizer using smart creation
    try:
        optimizer = create_smart_optimizer(model, optimizer_class, config)
        optimizer_info = f"{optimizer_class.__name__}"
        if hasattr(optimizer, 'primary_optimizer'):
            optimizer_info += f" (mixed with {type(optimizer.fallback_optimizer).__name__})"
    except Exception as e:
        raise AssertionError(f"Failed to create any working optimizer for {optimizer_class.__name__}: {e}")
    
    # Training loop
    model.train()
    num_epochs = 3
    total_steps = 0
    
    for epoch in range(num_epochs):
        for batch_X, batch_y in dl:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = loss_fn(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_steps += 1
    
    post_loss = measure_loss(model, dl)
    
    # Check for learning
    improvement_threshold = 0.8  # Require 10% improvement from initial
    if not (post_loss < pre_loss * improvement_threshold):
        raise AssertionError(
            f"Optimizer {optimizer_info} failed to learn: "
            f"pre_loss={pre_loss:.4f}, post_loss={post_loss:.4f}, "
            f"improvement={((pre_loss - post_loss) / pre_loss * 100):.1f}% "
            f"(required: {((1 - improvement_threshold) * 100):.1f}%)"
        )
    
    return {
        "pre_loss": pre_loss,
        "post_loss": post_loss,
        "loss_reduction": (pre_loss - post_loss) / pre_loss,
        "total_steps": total_steps,
        "optimizer_class": optimizer_class.__name__,
        "optimizer_info": optimizer_info,
        "config": config
    }


# Create test parameters - every discovered optimizer gets tested
optimizer_test_params = []
for optimizer_class, config in zip(all_optimizer_classes, all_optimizer_configs):
    test_id = f"{optimizer_class.__name__}"
    if config and config.optimizer_kwargs:
        test_id += f"_{'-'.join(f'{k}={v}' for k, v in config.optimizer_kwargs.items())}"
    else:
        test_id += "_auto"
        
    optimizer_test_params.append(
        pytest.param(optimizer_class, config, id=test_id)
    )

# Handle case where no optimizers are found
if len(optimizer_test_params) == 0:
    print("[WARNING] No optimizer classes found in catalog/")
    # Add a dummy test to prevent pytest from failing
    optimizer_test_params.append(
        pytest.param(None, None, id="no_optimizers_found")
    )


@pytest.mark.parametrize("optimizer_class,config", optimizer_test_params)
def test_universal_optimizer_learning(optimizer_class, config):
    """
    Test that each discovered optimizer can learn a simple task.
    Uses smart creation to handle constraints automatically.
    """
    if optimizer_class is None:
        pytest.fail("No optimizer classes found. Check the catalog/ directory.")
    
    universal_optimizer_test(optimizer_class, config)
