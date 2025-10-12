import copy

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from gpt_lab.train_loops.base_loop import run_training as run_training_base_loop
from gpt_lab.train_loops.grad_accum import run_training
from gpt_lab.train_loops.tests.test_utils import SimpleTestTrainingModel, AVAILABLE_DEVICES


@pytest.mark.parametrize("run_training_fn,device", [(run_training, device) for device in AVAILABLE_DEVICES])
def test_accumulation_correctness(run_training_fn, device):
    """Test that grad accumulation matches equivalent large batch training."""
    torch.manual_seed(0)
    X = torch.randn(16, 32).to(device)
    y = (X.sum(dim=1) > 0).long().to(device)
    ds = TensorDataset(X, y)

    # Use shuffle=False to ensure consistent sample ordering between different batch sizes
    # With shuffle=True, different batch sizes produce different shuffle orders even with same seed
    dl1 = DataLoader(ds, batch_size=8, shuffle=False)
    dl2 = DataLoader(ds, batch_size=4, shuffle=False)

    backbone1 = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 2))
    backbone2 = copy.deepcopy(backbone1)
    loss_fn = nn.CrossEntropyLoss()
    model1 = SimpleTestTrainingModel(backbone1, loss_fn)
    model2 = SimpleTestTrainingModel(backbone2, loss_fn)
    optim1 = torch.optim.AdamW(model1.parameters(), lr=3e-3)
    optim2 = torch.optim.AdamW(model2.parameters(), lr=3e-3)

    model1.to(device)
    model2.to(device)

    params1 = model1.state_dict()
    params2 = model2.state_dict()
    for key in params1:
        assert torch.allclose(params1[key], params2[key], atol=1e-6), f"Mismatch in parameter: {key}"

    result1 = run_training_base_loop(
        model=model1,
        optimizer=optim1,
        train_loader=dl1,
    )
    result2 = run_training_fn(
        model=model2,
        optimizer=optim2,
        train_loader=dl2,
        accum_steps=2,
    )

    final_params1 = result1['model'].state_dict()
    final_params2 = result2['model'].state_dict()
    for key in final_params1:
        assert torch.allclose(final_params1[key], final_params2[key], atol=1e-2), f"Mismatch in parameter: {key}"


# Registry for discovery by catalog_test.py and catalog_llm_compiler.py
__specific_tests__ = [
    test_accumulation_correctness,
]