# Train Loops Catalog

The train loops catalog provides atomic features that compose to create custom training loops.

## Smart API

The main entry point is `smart_train()`, which automatically selects and composes atomic features based on the arguments you provide:

```python
from gpt_lab.train_loops import smart_train

smart_train(
    model=model,
    train_loader=train_loader,
    optimizer=optimizer,
    # Atomic features are automatically composed
    num_epochs=10,
    grad_accum_steps=4,
    mixed_precision=True,
    grad_clip=1.0,
    lr_schedule="cosine",
    checkpoint_every_n_epochs=1
)
```

**How it works:**
- Analyzes kwargs to determine which atomic features are needed
- If single feature: executes directly
- If multiple features: uses LLM to compile an optimized combined loop
- Validates and caches compiled loops for reuse

## Available Atomic Features

### Gradient Management

**Gradient Accumulation** (`grad_accum.py`):
- `grad_accum_steps: int = 1`

**Gradient Clipping** (`grad_norm_clip.py`, `elem_grad_clip.py`):
- `grad_clip: float = None` - Norm-based clipping
- `elem_grad_clip: float = None` - Element-wise clipping

**Gradient Monitoring** (`gradient_monitoring.py`):
- `log_grad_stats: bool = False`

### Training Control

**Multi-Epoch Training** (`multi_epoch.py`):
- `num_epochs: int`

**Step Limiting** (`step_limiting.py`):
- `max_steps: int = None`

**Early Stopping** (`early_stopping.py`):
- `early_stop_patience: int = None`
- `early_stop_min_delta: float = 0.0`

### Validation

**Validation Loop** (`validation.py`):
- `val_loader: DataLoader = None`
- `validate_every_n_epochs: int = 1`

### Optimization

**Mixed Precision** (`mixed_precision.py`):
- `mixed_precision: bool = False`

**Learning Rate Scheduling** (`lr_scheduling.py`):
- `lr_schedule: str = None` - Options: "cosine", "linear", "step"
- `warmup_steps: int = 0`

### Checkpointing

**Epoch-based** (`checkpoint_over_epochs.py`):
- `checkpoint_every_n_epochs: int = None`

**Step-based** (`checkpoint_over_steps.py`):
- `checkpoint_every_n_steps: int = None`

**Best Model** (`checkpoint_best_model.py`):
- `save_best_model: bool = False`
- `best_model_metric: str = "val_loss"`

### Monitoring

**Progress Bars** (`tqdm.py`):
- `use_tqdm: bool = False`

**Loss Tracking** (`loss_tracking.py`):
- Automatically enabled

**Logging** (`logging.py`):
- `log_every_n_steps: int = 10`

## Adding Custom Features

Create atomic features in your experiment's `gpt_lab/train_loops/` directory:

```python
# experiments/my_exp/gpt_lab/train_loops/my_feature.py

def my_custom_feature(model, optimizer, train_loader, my_param=None, **kwargs):
    """
    Custom training loop feature.
    
    Args:
        my_param: Description of parameter
    """
    for batch in train_loader:
        # Your training logic
        pass
    
    return {"status": "complete"}
```

Then use it with `smart_train()`:

```python
smart_train(
    model=model,
    train_loader=train_loader,
    optimizer=optimizer,
    my_param=some_value  # Activates your custom feature
)
```

## Testing

Train loop features are tested automatically:

```bash
pytest tests/src/gpt_lab/train_loops/ -v
```

Features can also have co-located tests that automatically run on compiled loops:

```python
# catalogs/core/gpt_lab/train_loops/test_my_feature.py
def test_my_feature(run_training_fn, device):
    # Test implementation
    pass

__specific_tests__ = [test_my_feature]
```

## Contributing

See `catalogs/core/gpt_lab/train_loops/` for examples of atomic features. Each feature should:
- Accept standard args (model, optimizer, train_loader)
- Accept feature-specific kwargs
- Return a dictionary with results
- Be composable with other features
- Meet the criteria laid out in all catalog tests located in `tests/src/gpt_lab/train_loops/test_train_loops_catalog.py`