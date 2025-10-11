from typing import Optional, Dict, Any

import torch
import torch.nn as nn


def run_training(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader,
    *,
    # gradient noise knobs
    grad_noise_std: Optional[float] = None,
    grad_noise_decay: float = 0.99,
    # misc
    **kwargs,
) -> Dict[str, Any]:
    """Atomic training loop demonstrating gradient noise injection for regularization."""
    model.train()
    
    current_noise_std = grad_noise_std
    
    for batch in train_loader:
        loss = model(batch)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        
        # Add gradient noise if enabled
        if grad_noise_std is not None and current_noise_std > 0:
            with torch.no_grad():
                for param in model.parameters():
                    if param.grad is not None:
                        noise = torch.randn_like(param.grad) * current_noise_std
                        param.grad.add_(noise)
            
            # Decay noise std
            current_noise_std *= grad_noise_decay

        optimizer.step()

    return {"model": model}
