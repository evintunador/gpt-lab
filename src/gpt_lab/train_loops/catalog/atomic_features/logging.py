import logging
from typing import Dict, Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def run_training(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader,
    *,
    # logging
    enable_logging: bool = False,
    # misc
    **kwargs,
) -> Dict[str, Any]:
    """
    Atomic training loop demonstrating how to integrate standard Python logging for 
    both diagnostics and structured metrics.

    LLM Compiler Hint: If metric logging is enabled, use the standard `logging` module
    to log all relevant metrics. Log metrics using logger.info() with an `extra`
    dictionary for structured data. For example:
    logger.info("Training step", extra={"metrics": {"train_loss": loss.item()}})
    If a feature like `validation` is present, also log its outputs (e.g., `val_loss`)
    when they are computed.
    """
    model.train()

    for batch in train_loader:
        loss = model(batch)

        if enable_logging:
            logger.info("Training step", extra={"metrics": {"train_loss": loss.item()}})

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    return {"model": model}
