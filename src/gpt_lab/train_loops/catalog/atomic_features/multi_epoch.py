from typing import Optional, Dict, Any

import torch
import torch.nn as nn


def run_training(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader,
    *,
    # multi-epoch knobs
    num_epochs: int = 1,
    # misc
    **kwargs,
) -> Dict[str, Any]:
    """Atomic training loop demonstrating multi-epoch training."""
    model.train()
    
    for epoch in range(num_epochs):
        for batch in train_loader:
            loss = model(batch)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    return {"model": model}