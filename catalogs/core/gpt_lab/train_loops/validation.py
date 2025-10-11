from typing import Dict, Any, List

import torch
import torch.nn as nn


@torch.no_grad()
def _eval_loss(model: nn.Module, loader) -> float:
    """Helper to compute validation loss."""
    was_training = model.training
    model.eval()
    total, count = 0.0, 0
    for batch in loader:
        loss = model(batch)
        total += float(loss.detach().cpu().item())
        count += 1
    if was_training:
        model.train()
    return total / max(count, 1)


def run_training(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader,
    *,
    # validation knobs
    val_loader = None,
    val_interval: int = 10,
    # misc
    **kwargs,
) -> Dict[str, Any]:
    """Atomic training loop demonstrating validation during training."""
    model.train()
    
    val_loss_history: List[float] = []
    
    step_count = 0
    for batch in train_loader:
        loss = model(batch)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
        if val_loader is not None and step_count > 0 and step_count % val_interval == 0:
            val_loss = _eval_loss(model, val_loader)
            val_loss_history.append(val_loss)
        step_count += 1
    
    # Final validation at end of training
    if val_loader is not None and (step_count == 0 or step_count % val_interval != 0):
        final_val_loss = _eval_loss(model, val_loader)
        val_loss_history.append(final_val_loss)

    result = {"model": model}
    if val_loader is not None:
        result["val_loss_history"] = val_loss_history
    
    return result