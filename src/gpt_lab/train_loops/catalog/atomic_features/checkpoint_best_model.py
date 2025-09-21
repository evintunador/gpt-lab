from typing import Optional, Dict, Any
import os

import torch
import torch.nn as nn

import gpt_lab.checkpointer as checkpointer


@torch.no_grad()
def _eval_loss(model: nn.Module, loss_fn, loader) -> float:
    """Helper to compute validation loss."""
    was_training = model.training
    model.eval()
    total, count = 0.0, 0
    for batch in loader:
        xb, yb = batch
        logits = model(xb)
        loss = loss_fn(logits, yb)
        total += float(loss.detach().cpu().item())
        count += 1
    if was_training:
        model.train()
    return total / max(count, 1)


def run_training(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn,
    train_loader,
    *,
    # feature knobs
    save_best_model: bool = False,
    output_dir: Optional[str] = None,
    val_loader=None,
    val_interval: int = 10,
    # misc
    **kwargs,
) -> Dict[str, Any]:
    """Atomic feature for saving the best model based on validation loss."""
    model.train()
    best_val_loss = float('inf')
    result = {"model": model}

    optimizer.zero_grad(set_to_none=True)
    
    for batch_idx, batch in enumerate(train_loader):
        xb, yb = batch
        logits = model(xb)
        loss = loss_fn(logits, yb)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        
        if save_best_model:
            if val_loader is None or output_dir is None:
                raise ValueError("val_loader and output_dir must be provided when save_best_model is True.")

            if (batch_idx % val_interval == 0) or (batch_idx == len(train_loader) - 1):
                val_loss = _eval_loss(model, loss_fn, val_loader)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    result['best_val_loss'] = best_val_loss
                    raw_model = model.module if hasattr(model, 'module') else model
                    checkpointer.save_checkpoint(
                        save_dir=os.path.join(output_dir, "checkpoints"),
                        filename="best_model.pt",
                        # include ALL metadata that would be required to resume training from this epoch
                        metadata={"val_loss": best_val_loss, "step": batch_idx, "config": kwargs.get("config", {})},
                        # include ALL objects with a state_dict as kwargs
                        model=raw_model,
                        optimizer=optimizer,
                    )

    return result
