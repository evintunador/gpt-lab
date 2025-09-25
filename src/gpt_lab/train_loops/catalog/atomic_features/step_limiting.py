from typing import Optional, Dict, Any

import torch
import torch.nn as nn


def run_training(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn,
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

    data_iter = iter(train_loader)
    step_count = 0

    while True:
        if total_steps is not None and step_count >= total_steps:
            break

        try:
            batch = next(data_iter)
        except StopIteration:
            if total_steps is None:
                break
            data_iter = iter(train_loader)
            batch = next(data_iter)

        xb, yb = batch
        logits = model(xb)
        loss = loss_fn(logits, yb)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        step_count += 1

    return {"model": model}