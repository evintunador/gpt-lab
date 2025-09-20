from typing import Optional, Dict, Any, List
import torch
import torch.nn as nn


def run_training(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn,
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
    
    for batch_idx, batch in enumerate(train_loader):
        xb, yb = batch
        logits = model(xb)
        loss = loss_fn(logits, yb)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
        # Log learning rate if enabled
        if log_lr_changes and (batch_idx % lr_log_interval == 0):
            # Get current learning rate from first parameter group
            current_lr = optimizer.param_groups[0]['lr']
            lr_history.append(current_lr)

    result = {"model": model}
    if log_lr_changes and lr_history:
        result["lr_history"] = lr_history
    
    return result
