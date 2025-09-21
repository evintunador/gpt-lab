from typing import Optional, Dict, Any
import os
import torch
import torch.nn as nn
import checkpointer

# Metadata for smart_train
__smart_train_metadata__ = {
    "conflicts_with": ["checkpoint_over_steps"],
}

def run_training(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn,
    train_loader,
    *,
    # feature knobs
    save_every_epochs: Optional[int] = None,
    output_dir: Optional[str] = None,
    num_epochs: int = 1,
    # misc
    **kwargs,
) -> Dict[str, Any]:
    """Atomic feature for saving checkpoints every N epochs."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    
    for epoch in range(num_epochs):
        for batch in train_loader:
            xb, yb = batch
            logits = model(xb)
            loss = loss_fn(logits, yb)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        if (save_every_epochs is not None 
            and output_dir is not None
            and (epoch == 0 
                or epoch % save_every_epochs == 0 
                or epoch == num_epochs - 1)):
            raw_model = model.module if hasattr(model, 'module') else model
            checkpointer.save_checkpoint(
                save_dir=os.path.join(output_dir, "checkpoints"),
                filename=f"epoch_{epoch}.pt",
                # include ALL metadata that would be required to resume training from this epoch
                metadata={"epoch": epoch, "config": kwargs.get("config", {})}, 
                # include ALL objects with a state_dict as kwargs
                model=raw_model,
                optimizer=optimizer,
            )

    return {"model": model}
