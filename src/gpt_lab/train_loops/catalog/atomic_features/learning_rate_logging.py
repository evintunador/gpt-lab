from typing import Optional, Dict, Any, List

import torch
import torch.nn as nn


def run_training(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader,
    *,
    # learning rate logging knobs
    log_lr_changes: bool = False,
    lr_log_interval: int = 1,
    # misc
    **kwargs,
) -> Dict[str, Any]:
    """Atomic training loop demonstrating learning rate logging."""
    model.train()
    
    lr_history: List[float] = []
    step_count = 0

    for batch in train_loader:
        loss = model(batch)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
        # Log learning rate if enabled
        if log_lr_changes and (step_count % lr_log_interval == 0):
            # Get current learning rate from first parameter group
            current_lr = optimizer.param_groups[0]['lr']
            lr_history.append(current_lr)

        step_count += 1

    result = {"model": model}
    if log_lr_changes and lr_history:
        result["lr_history"] = lr_history
    
    return result
