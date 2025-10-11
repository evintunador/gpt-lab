from typing import Optional, Dict, Any, List

import torch
import torch.nn as nn


def run_training(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader,
    *,
    # weight monitoring knobs
    track_weight_norms: bool = False,
    track_weight_changes: bool = False,
    weight_log_interval: int = 1,
    # misc
    **kwargs,
) -> Dict[str, Any]:
    """Atomic training loop demonstrating weight monitoring."""
    model.train()
    
    weight_norm_history: List[float] = []
    weight_change_history: List[float] = []
    step_count = 0
    
    # Store initial weights if tracking changes
    initial_weights = {}
    if track_weight_changes:
        with torch.no_grad():
            for name, param in model.named_parameters():
                initial_weights[name] = param.data.clone()
    
    for batch in train_loader:
        loss = model(batch)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
        # Monitor weights if enabled
        if (track_weight_norms or track_weight_changes) and (step_count % weight_log_interval == 0):
            with torch.no_grad():
                if track_weight_norms:
                    total_norm = 0.0
                    param_count = 0
                    for param in model.parameters():
                        if param.data is not None:
                            param_norm = param.data.norm()
                            total_norm += param_norm.item() ** 2
                            param_count += 1
                    
                    if param_count > 0:
                        total_norm = total_norm ** 0.5
                        weight_norm_history.append(total_norm)
                
                if track_weight_changes and initial_weights:
                    total_change = 0.0
                    param_count = 0
                    for name, param in model.named_parameters():
                        if name in initial_weights:
                            change = (param.data - initial_weights[name]).norm()
                            total_change += change.item() ** 2
                            param_count += 1
                    
                    if param_count > 0:
                        total_change = total_change ** 0.5
                        weight_change_history.append(total_change)
        
        step_count += 1

    result = {"model": model}
    if track_weight_norms and weight_norm_history:
        result["weight_norm_history"] = weight_norm_history
    if track_weight_changes and weight_change_history:
        result["weight_change_history"] = weight_change_history
    
    return result
