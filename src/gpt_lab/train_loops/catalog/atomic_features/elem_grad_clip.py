from typing import Optional, Dict, Any

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_value_


def run_training(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader,
    *,
    # elementwise grad clipping knobs
    elem_grad_clip: Optional[float] = None,
    # misc
    **kwargs,
) -> Dict[str, Any]:
    """Atomic training loop demonstrating elementwise gradient clipping."""
    model.train()

    for batch in train_loader:
        loss = model(batch)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # elementwise gradient clipping
        if elem_grad_clip is not None:
            params = [p for p in model.parameters() if p.grad is not None]
            if params:
                clip_grad_value_(params, elem_grad_clip)

        optimizer.step()

    return {"model": model}