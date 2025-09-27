# TODO: update this atomic feature to comply with latest pytorch which i think does allow amp on mps
from typing import Optional, Dict, Any

import torch
import torch.nn as nn


def _is_amp_compatible(device_str: str) -> bool:
    """Check if device supports AMP properly."""
    if device_str.startswith('cuda'):
        return True
    elif device_str == 'mps':
        # MPS has issues with GradScaler in certain PyTorch versions
        return False
    return False


def run_training(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader,
    *,
    # mixed precision knobs
    use_amp: Optional[bool] = False,  # False = disabled by default
    loss_scale: Optional[float] = None,
    device: Optional[str] = None,
    # misc
    **kwargs,
) -> Dict[str, Any]:
    """Atomic training loop demonstrating automatic mixed precision (AMP)."""
    model.train()
    
    # Infer device from model if not provided
    if device is None:
        device = str(next(model.parameters()).device)
    
    # Auto-detect AMP compatibility if not explicitly set
    if use_amp is None:
        use_amp = _is_amp_compatible(device)
    
    # Force disable AMP on incompatible devices
    if use_amp and not _is_amp_compatible(device):
        print(f"Warning: AMP requested but not compatible with {device}. Falling back to FP32.")
        use_amp = False
    
    # Create GradScaler for AMP if enabled and compatible
    scaler = None
    if use_amp:
        try:
            scaler = torch.amp.GradScaler(device)
        except Exception as e:
            print(f"Warning: Failed to create GradScaler for {device}: {e}. Falling back to FP32.")
            use_amp = False
    
    for batch in train_loader:
        optimizer.zero_grad(set_to_none=True)
        
        if use_amp and scaler is not None:
            # Forward pass with autocast
            with torch.amp.autocast(device):
                loss = model(batch)
            
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard precision training
            loss = model(batch)
            loss.backward()
            optimizer.step()

    return {"model": model, "used_amp": use_amp}