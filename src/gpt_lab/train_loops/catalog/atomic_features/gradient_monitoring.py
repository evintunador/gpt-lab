from typing import Optional, Dict, Any, List

import torch
import torch.nn as nn


def run_training(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader,
    *,
    # gradient monitoring knobs
    track_grad_norms: bool = False,
    track_grad_flow: bool = False,
    grad_log_interval: int = 1,
    # misc
    **kwargs,
) -> Dict[str, Any]:
    """Atomic training loop demonstrating gradient monitoring."""
    model.train()
    
    grad_norm_history: List[float] = []
    grad_flow_history: List[Dict[str, float]] = []
    step_count = 0

    for batch in train_loader:
        loss = model(batch)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        
        # Monitor gradients if enabled
        if (track_grad_norms or track_grad_flow) and (step_count % grad_log_interval == 0):
            with torch.no_grad():
                if track_grad_norms:
                    total_norm = 0.0
                    param_count = 0
                    for param in model.parameters():
                        if param.grad is not None:
                            param_norm = param.grad.norm()
                            total_norm += param_norm.item() ** 2
                            param_count += 1
                    
                    if param_count > 0:
                        total_norm = total_norm ** 0.5
                        grad_norm_history.append(total_norm)
                
                if track_grad_flow:
                    grad_flow = {}
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            grad_flow[name] = param.grad.norm().item()
                    
                    if grad_flow:
                        grad_flow_history.append(grad_flow)

        optimizer.step()
        step_count += 1

    result = {"model": model}
    if track_grad_norms and grad_norm_history:
        result["grad_norm_history"] = grad_norm_history
    if track_grad_flow and grad_flow_history:
        result["grad_flow_history"] = grad_flow_history
    
    return result
