from typing import Optional, Dict, Any

import torch
import torch.nn as nn

from logger import ExperimentLogger


def run_training(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn,
    train_loader,
    *,
    # logging
    logger: Optional[ExperimentLogger] = None,
    # misc
    **kwargs,
) -> Dict[str, Any]:
    """
    Atomic training loop demonstrating how to integrate the ExperimentLogger.

    LLM Compiler Hint: If a `logger` object is provided, use it to log
    all relevant metrics and outputs from other combined features. For example,
    log the `loss` at each step, and if a feature like `validation` is present,
    log its outputs (e.g., `val_loss`) when they are computed.
    """
    model.train()

    for batch_idx, batch in enumerate(train_loader):
        xb, yb = batch
        logits = model(xb)
        loss = loss_fn(logits, yb)

        if logger:
            logger.log({"train_loss": loss.item()})

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    return {"model": model}
