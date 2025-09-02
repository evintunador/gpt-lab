from typing import Optional, Dict, Any
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_


def run_training(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn,
    train_loader,
    *,
    # grad accum knobs
    accum_steps: int = 1,
    # grad norm clipping knobs
    norm_clip_value: Optional[float] = None,
    # step limiting knobs
    max_steps: Optional[int] = None,
    # learning rate scheduling knobs
    lr_scheduler_type: str = "none",
    warmup_steps: int = 0,
    max_lr: Optional[float] = None,
    min_lr: float = 0.0,
    total_steps: Optional[int] = None,
    # mixed precision knobs
    use_amp: bool = False,
    # misc
    **kwargs,
) -> Dict[str, Any]:
    """Combined training loop with gradient accumulation, gradient clipping, step limiting, LR scheduling, and mixed precision."""
    model.train()

    # Validate accum_steps
    if accum_steps is None or accum_steps < 1:
        accum_steps = 1

    # Determine total steps for scheduling
    if total_steps is None:
        if max_steps is not None:
            total_steps = max_steps
        else:
            try:
                total_steps = len(train_loader)
            except:
                total_steps = 1000  # fallback

    # Get current learning rate for max_lr if not specified
    if max_lr is None:
        max_lr = optimizer.param_groups[0]['lr']

    # Create scheduler
    scheduler = None
    if lr_scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps, eta_min=min_lr
        )
    elif lr_scheduler_type == "linear":
        def lambda_lr(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return max(min_lr / max_lr, 1.0 - progress)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda_lr)
    elif lr_scheduler_type == "step":
        step_size = max(1, total_steps // 3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)
    elif lr_scheduler_type == "exponential":
        gamma = (min_lr / max_lr) ** (1.0 / total_steps)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    # Create GradScaler for AMP if enabled
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    # Initialize counters and state
    micro_idx = 0
    step_count = 0
    optimizer.zero_grad(set_to_none=True)

    # Set up data iteration based on step limiting
    if max_steps is None:
        # No step limit, just train through the loader once
        data_iter = iter(train_loader)
        def get_next_batch():
            try:
                return next(data_iter)
            except StopIteration:
                return None
    else:
        # Step limiting with data cycling
        data_iter = iter(train_loader)
        def get_next_batch():
            nonlocal data_iter
            try:
                return next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                return next(data_iter)

    # Main training loop
    while True:
        if max_steps is not None and step_count >= max_steps:
            break

        batch = get_next_batch()
        if batch is None:
            break

        xb, yb = batch

        if use_amp:
            # Forward pass with autocast
            with torch.cuda.amp.autocast():
                logits = model(xb)
                loss = loss_fn(logits, yb)

            if accum_steps > 1:
                loss = loss / float(accum_steps)

            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
        else:
            # Standard precision training
            logits = model(xb)
            loss = loss_fn(logits, yb)

            if accum_steps > 1:
                loss = loss / float(accum_steps)

            loss.backward()

        micro_idx += 1

        # Check if we should take an optimizer step
        should_step = (micro_idx % accum_steps == 0)

        if should_step:
            if use_amp:
                # Apply gradient clipping if specified
                if norm_clip_value is not None:
                    scaler.unscale_(optimizer)
                    params = [p for p in model.parameters() if p.grad is not None]
                    if params:
                        clip_grad_norm_(params, norm_clip_value, norm_type=2.0)

                scaler.step(optimizer)
                scaler.update()
            else:
                # Apply gradient clipping if specified
                if norm_clip_value is not None:
                    params = [p for p in model.parameters() if p.grad is not None]
                    if params:
                        clip_grad_norm_(params, norm_clip_value, norm_type=2.0)

                optimizer.step()

            optimizer.zero_grad(set_to_none=True)

            # Step scheduler if it exists
            if scheduler is not None:
                scheduler.step()

            step_count += 1

    # Handle case where last accumulation window is incomplete
    if micro_idx % accum_steps != 0:
        if use_amp:
            if norm_clip_value is not None:
                scaler.unscale_(optimizer)
                params = [p for p in model.parameters() if p.grad is not None]
                if params:
                    clip_grad_norm_(params, norm_clip_value, norm_type=2.0)

            scaler.step(optimizer)
            scaler.update()
        else:
            if norm_clip_value is not None:
                params = [p for p in model.parameters() if p.grad is not None]
                if params:
                    clip_grad_norm_(params, norm_clip_value, norm_type=2.0)

            optimizer.step()

    return {"model": model}