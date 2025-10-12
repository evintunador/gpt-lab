from typing import Optional, Dict, Any

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
    # early stopping knobs
    val_loader = None,
    patience: int = 5,
    min_delta: float = 0.0,
    val_interval: int = 10,
    # misc
    **kwargs,
) -> Dict[str, Any]:
    """Atomic training loop demonstrating early stopping based on validation loss."""
    model.train()
    
    if val_loader is None:
        # No validation loader, just train normally
        for batch in train_loader:
            loss = model(batch)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        return {"model": model}
    
    best_val_loss = float('inf')
    patience_counter = 0
    step_count = 0

    for batch in train_loader:
        loss = model(batch)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
        # Check for early stopping
        if step_count % val_interval == 0:
            val_loss = _eval_loss(model, val_loader)
            
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"Early stopping at batch {step_count}")
                break

    return {"model": model}