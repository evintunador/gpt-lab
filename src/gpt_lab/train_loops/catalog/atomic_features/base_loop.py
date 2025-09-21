from typing import Optional, Dict, Any
import torch
import torch.nn as nn


def run_training(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn,
    train_loader,
    #*, # feature-specific arguments should follow *
    **kwargs,
) -> Dict[str, Any]:
    """A base 'zero feature' training loop to build your new atomic feature loop off of."""
    model.train()
    optimizer.zero_grad(set_to_none=True)

    for batch in train_loader:
        xb, yb = batch
        logits = model(xb)
        loss = loss_fn(logits, yb)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    return {"model": model}