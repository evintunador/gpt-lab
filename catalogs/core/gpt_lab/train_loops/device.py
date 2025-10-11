from typing import Optional, Dict, Any
import torch
import torch.nn as nn

# to_device handles tensors, modules, lists, tuples, and dictionary values and does so recursively
from gpt_lab.device import to_device 


def run_training(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader,
    *, 
    # device knobs
    device: Optional[str] = None,
    # misc
    **kwargs,
) -> Dict[str, Any]:
    """
    Atomic training loop demonstrating device placement and data movement.
    If other atomic features introduce objects that need device placement,
    be sure to move them as well.
    """
    # Determine target device
    if device is None:
        device = str(next(model.parameters()).device)
    
    # Move model to device
    model = model.to(device)
    model.train()
    optimizer.zero_grad(set_to_none=True)

    for batch in train_loader:
        # Move batch data to device
        batch = to_device(batch, device)
        
        loss = model(batch)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    return {"model": model}