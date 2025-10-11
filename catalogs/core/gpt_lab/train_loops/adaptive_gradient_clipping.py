from typing import Optional, Dict, Any
import torch
import torch.nn as nn


def run_training(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader,
    *,
    # adaptive gradient clipping knobs
    agc_clip_factor: Optional[float] = None,
    agc_eps: float = 1e-3,
    # misc
    **kwargs,
) -> Dict[str, Any]:
    """Atomic training loop demonstrating Adaptive Gradient Clipping (AGC)."""
    model.train()

    for batch in train_loader:
        loss = model(batch)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # Apply adaptive gradient clipping if enabled
        if agc_clip_factor is not None:
            with torch.no_grad():
                for param in model.parameters():
                    if param.grad is not None:
                        # Calculate parameter norm and gradient norm
                        param_norm = param.norm(dim=None, keepdim=False)
                        grad_norm = param.grad.norm(dim=None, keepdim=False)
                        
                        # Calculate maximum allowed gradient norm
                        max_norm = agc_clip_factor * torch.max(param_norm, torch.tensor(agc_eps, device=param.device))
                        
                        # Clip gradient if it exceeds the max norm
                        if grad_norm > max_norm:
                            param.grad.mul_(max_norm / (grad_norm + 1e-6))

        optimizer.step()

    return {"model": model}
