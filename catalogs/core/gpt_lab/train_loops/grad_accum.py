from typing import Dict, Any

import torch
import torch.nn as nn


def run_training(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader,
    *,
    # grad accum knobs
    accum_steps: int = 1,
    # misc
    **kwargs,
) -> Dict[str, Any]:
    """Atomic training loop demonstrating gradient accumulation."""
    model.train()
    optimizer.zero_grad(set_to_none=True)

    if accum_steps is None or accum_steps < 1:
        accum_steps = 1

    step_count = 0
    micro_idx = 0
    for batch in train_loader:
        loss = model(batch)

        if accum_steps > 1:
            loss = loss / float(accum_steps)

        loss.backward()

        micro_idx += 1

        if micro_idx % accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step_count += 1

    # Handle case where last accumulation window is incomplete
    if micro_idx % accum_steps != 0:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        step_count += 1

    return {"model": model}