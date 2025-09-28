from typing import Optional, Dict, Any

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_


def run_training(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader,
    *,
    # grad norm clipping knobs
    norm_clip_value: Optional[float] = None,
    # misc
    **kwargs,
) -> Dict[str, Any]:
    """Atomic training loop demonstrating gradient norm clipping."""
    model.train()

    for batch in train_loader:
        loss = model(batch)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # gradient clipping
        if norm_clip_value is not None:
            params = [p for p in model.parameters() if p.grad is not None]
            if params:
                clip_grad_norm_(params, norm_clip_value, norm_type=2.0)

        optimizer.step()

    return {"model": model}


# Optional metadata for smart_train conflict detection
__smart_train_metadata__ = {
    "conflicts_with": ["elem_grad_clip"],  # Cannot use both gradient clipping methods
}