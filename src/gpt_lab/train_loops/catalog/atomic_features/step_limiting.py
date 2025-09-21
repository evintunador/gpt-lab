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
    max_steps: Optional[int] = None,
    # misc
    **kwargs,
) -> Dict[str, Any]:
    """Atomic training loop demonstrating step limiting with infinite data cycling."""
    model.train()
    
    if max_steps is None:
        # No step limit, just train through the loader once
        for batch in train_loader:
            xb, yb = batch
            logits = model(xb)
            loss = loss_fn(logits, yb)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    else:
        # Step limiting with data cycling
        data_iter = iter(train_loader)
        step_count = 0
        
        while step_count < max_steps:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)
            
            xb, yb = batch
            logits = model(xb)
            loss = loss_fn(logits, yb)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            
            step_count += 1

    return {"model": model}