from typing import Optional, Dict, Any

import torch
import torch.nn as nn


def run_training(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader,
    *,
    # gradient skipping knobs
    grad_skip_threshold: Optional[float] = None,
    skip_strategy: str = "norm",  # "norm" or "inf" or "nan_inf"
    # misc
    **kwargs,
) -> Dict[str, Any]:
    """Atomic training loop demonstrating gradient skipping for problematic gradients."""
    model.train()
    
    skipped_steps = 0
    total_steps = 0

    for batch in train_loader:
        loss = model(batch)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        
        # Check if we should skip this gradient update
        should_skip = False
        total_steps += 1
        
        if grad_skip_threshold is not None:
            with torch.no_grad():
                if skip_strategy == "norm":
                    # Calculate total gradient norm
                    total_norm = 0.0
                    for param in model.parameters():
                        if param.grad is not None:
                            param_norm = param.grad.norm()
                            total_norm += param_norm.item() ** 2
                    total_norm = total_norm ** 0.5
                    
                    if total_norm > grad_skip_threshold:
                        should_skip = True
                        
                elif skip_strategy == "inf":
                    # Check for infinite gradients
                    for param in model.parameters():
                        if param.grad is not None and torch.isinf(param.grad).any():
                            should_skip = True
                            break
                            
                elif skip_strategy == "nan_inf":
                    # Check for NaN or infinite gradients
                    for param in model.parameters():
                        if param.grad is not None:
                            if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                                should_skip = True
                                break
        
        if should_skip:
            skipped_steps += 1
            # Zero gradients without stepping optimizer
            optimizer.zero_grad(set_to_none=True)
        else:
            optimizer.step()

    result = {"model": model}
    if grad_skip_threshold is not None:
        result["skipped_steps"] = skipped_steps
        result["total_steps"] = total_steps
    
    return result
