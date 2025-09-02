from typing import Optional, Dict, Any
import torch
import torch.nn as nn


def run_training(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn,
    train_loader,
    *,
    # mixed precision knobs
    use_amp: bool = True,
    loss_scale: Optional[float] = None,
    # misc
    **kwargs,
) -> Dict[str, Any]:
    """Atomic training loop demonstrating automatic mixed precision (AMP)."""
    model.train()
    
    # Create GradScaler for AMP if enabled
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    
    for batch in train_loader:
        xb, yb = batch
        
        optimizer.zero_grad(set_to_none=True)
        
        if use_amp:
            # Forward pass with autocast
            with torch.cuda.amp.autocast():
                logits = model(xb)
                loss = loss_fn(logits, yb)
            
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard precision training
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()

    return {"model": model}