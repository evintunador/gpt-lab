from typing import Optional, Dict, Any
import os

import torch
import torch.nn as nn

import gpt_lab.checkpointer as checkpointer


# Metadata for smart_train
__smart_train_metadata__ = {
    "conflicts_with": ["checkpoint_over_epochs"],
}


def run_training(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn,
    train_loader,
    *,
    # feature knobs
    save_every_steps: Optional[int] = None,
    output_dir: Optional[str] = None,
    # misc
    **kwargs,
) -> Dict[str, Any]:
    """Atomic feature for saving checkpoints every N steps."""
    model.train()
    optimizer.zero_grad(set_to_none=True)

    step_count = 0
    for batch in train_loader:
        xb, yb = batch
        logits = model(xb)
        loss = loss_fn(logits, yb)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        
        step_count += 1

        if (save_every_steps is not None 
            and output_dir is not None
            and step_count > 0 
            and step_count % save_every_steps == 0):
                raw_model = model.module if hasattr(model, 'module') else model
                checkpointer.save_checkpoint(
                    save_dir=os.path.join(output_dir, "checkpoints"),
                    filename=f"step_{step_count}.pt",
                    # include ALL metadata that would be required to resume training from this epoch
                    metadata={"step": step_count, "config": kwargs.get("config", {})},
                    # include ALL objects with a state_dict as kwargs
                    model=raw_model,
                    optimizer=optimizer,
                )

    return {"model": model}
