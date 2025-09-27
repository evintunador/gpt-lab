from typing import Optional, Dict, Any

import torch
import torch.nn as nn


def run_training(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader,
    *,
    # learning rate scheduling knobs
    lr_scheduler_type: Optional[str] = None,
    scheduler_kwargs: Optional[Dict[str, Any]] = None,
    warmup_steps: int = 0,
    max_lr: Optional[float] = None,
    min_lr: float = 0.0,
    total_steps: Optional[int] = None,
    # misc
    **kwargs,
) -> Dict[str, Any]:
    """Atomic training loop demonstrating learning rate scheduling."""
    model.train()
    
    # Determine total steps if not provided
    if total_steps is None:
        try:
            total_steps = len(train_loader)
        except:
            total_steps = 1000  # fallback
    
    # Get current learning rate for max_lr if not specified
    if max_lr is None:
        max_lr = optimizer.param_groups[0]['lr']
    
    # Create scheduler
    scheduler = None
    scheduler_kwargs = scheduler_kwargs or {}
    if lr_scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps, eta_min=min_lr, **scheduler_kwargs
        )
    elif lr_scheduler_type == "linear":
        def lambda_lr(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return max(min_lr / max_lr, 1.0 - progress)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda_lr, **scheduler_kwargs)
    elif lr_scheduler_type == "step":
        step_size = max(1, total_steps // 3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1, **scheduler_kwargs)
    elif lr_scheduler_type == "exponential":
        gamma = (min_lr / max_lr) ** (1.0 / total_steps)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma, **scheduler_kwargs)
    elif lr_scheduler_type == "lambda_lr":
        if "lr_lambda" not in scheduler_kwargs:
            raise ValueError("`scheduler_kwargs` must contain `lr_lambda` function for `lambda_lr` scheduler.")
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, **scheduler_kwargs)
    
    step_count = 0
    for batch in train_loader:
        loss = model(batch)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
        # Step scheduler if it exists
        if scheduler is not None:
            scheduler.step()
        
        step_count += 1

    return {"model": model}
