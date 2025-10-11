from typing import Optional, Dict, Any

import torch
import torch.nn as nn


def run_training(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader,
    *,
    # step limiting knobs
    total_steps: Optional[int] = None,
    # misc
    **kwargs,
) -> Dict[str, Any]:
    """Atomic training loop demonstrating step limiting with infinite data cycling."""
    model.train()
    optimizer.zero_grad(set_to_none=True)

    step_count = 0
    should_break = False
    while not should_break:
        for batch in train_loader:
            if total_steps is not None and step_count >= total_steps:
                should_break = True
                break

            loss = model(batch)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            step_count += 1
        
        # For iterable-style datasets, we'll only do one pass if total_steps is not set
        if total_steps is None:
            should_break = True

    return {"model": model}