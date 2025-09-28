from typing import Optional, Dict, Any, List

import torch
import torch.nn as nn


def run_training(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader,
    *,
    # loss tracking knobs
    track_loss: bool = False,
    log_interval: int = 1,  # track every batch by default
    # misc
    **kwargs,
) -> Dict[str, Any]:
    """Atomic training loop demonstrating loss tracking."""
    model.train()
    
    train_loss_history: List[float] = []
    step_count = 0

    for batch in train_loader:
        loss = model(batch)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
        # Track loss if enabled
        if track_loss and (step_count % log_interval == 0):
            train_loss_history.append(float(loss.detach().cpu().item()))

        step_count += 1

    result = {"model": model}
    if track_loss:
        result["train_loss_history"] = train_loss_history
    
    return result