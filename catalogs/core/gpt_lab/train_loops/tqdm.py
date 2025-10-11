from typing import Dict, Any
import torch
import torch.nn as nn
from torch.utils.data import IterableDataset
from tqdm.auto import tqdm as tqdm_auto


def run_training(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader,
    *,
    # tqdm knobs
    use_tqdm: bool = False,
    # misc
    **kwargs,
) -> Dict[str, Any]:
    """A training loop with an optional tqdm progress bar."""
    model.train()
    optimizer.zero_grad(set_to_none=True)

    pbar = None
    is_map_style = not isinstance(train_loader.dataset, IterableDataset)
    if use_tqdm and is_map_style:
        pbar = tqdm_auto(desc="Training", leave=False, total=len(train_loader))

    try:
        for batch in train_loader:
            loss = model(batch)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            if pbar:
                pbar.update(1)
                pbar.set_postfix(loss=f"{loss.item():.4f}")
    finally:
        if pbar:
            pbar.close()

    return {"model": model}