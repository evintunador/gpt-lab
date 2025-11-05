# Checkpointer

The `gpt_lab.checkpointer` module provides flexible checkpoint saving and loading for reproducible ML experiments.

## Overview

The checkpointer component is designed to:
- Save and load model, optimizer, and any stateful PyTorch objects
- Store arbitrary metadata (metrics, epoch, step, git info, etc.)
- Work seamlessly with distributed training and `torch.compile`
- Provide simple, flexible APIs that work with any PyTorch object with `state_dict()`
- Automatically map loaded tensors to the correct device

## Core Functions

### `save_checkpoint()`

Saves a comprehensive training checkpoint:

```python
from gpt_lab.checkpointer import save_checkpoint

filepath = save_checkpoint(
    filepath="./checkpoints/epoch_10.pt",
    metadata={
        "epoch": 10,
        "step": 5000,
        "best_val_loss": 0.342,
        "git_info": repro_manager.git_info,
        "rng_states": repro_manager.get_rng_states()
    },
    model=model,
    optimizer=optimizer,
    scheduler=lr_scheduler
)
# Returns: "./checkpoints/epoch_10.pt"
```

**Parameters:**
- `filepath`: The full path, including directory and filename, for the checkpoint file. The directory will be created if it doesn't exist.
- `metadata`: Dictionary of arbitrary non-stateful metadata to save.
- `**stateful_objects`: Keyword arguments for any objects with a `state_dict()` method.

**Returns:** Full path to the saved checkpoint file.

**Checkpoint Structure:**
```python
{
    'metadata': {
        "epoch": 10,
        "rng_states": { ... } # If you added them
        # ...
    },
    'model': <model_state_dict>,
    'optimizer': <optimizer_state_dict>,
    # ... any other objects you passed
}
```

### `load_checkpoint()`

Loads a checkpoint and restores object states:

```python
from gpt_lab.checkpointer import load_checkpoint

# The model and optimizer can be on any device; the checkpointer will match it.
model = MyModel().to("cuda:0")
optimizer = torch.optim.Adam(model.parameters())

loaded_metadata = load_checkpoint(
    filepath="./checkpoints/epoch_10.pt",
    model=model,
    optimizer=optimizer,
    scheduler=lr_scheduler
)

# loaded_metadata contains all non-stateful info from the checkpoint, including the original metadata dict
epoch = loaded_metadata.get('epoch', 0)
step = loaded_metadata.get('step', 0)

# Optionally restore RNG states
rng_states = loaded_metadata.get('rng_states')
if rng_states:
    repro_manager.set_rng_states(rng_states)
```

**Parameters:**
- `filepath`: Path to the checkpoint file.
- `**stateful_objects`: Objects to load state into (must match keys from save).

**Returns:** A dictionary containing all non-state-dict data from the checkpoint (e.g., your `metadata` dictionary and any other non-stateful objects you saved).

**Behavior:**
- Calls `load_state_dict()` on each provided object in-place.
- **Automatic Device Mapping**: Automatically moves loaded tensors to the same device and dtype as the corresponding tensors in the provided `stateful_objects`. You no longer need to specify a `map_location`.
- Prints warnings for missing keys or objects.

## Best Practices

### 1. Always Include Git Info

Use a reproducibility manager to capture git information:

```python
from gpt_lab.reproducibility import ReproducibilityManager

repro = ReproducibilityManager()

save_checkpoint(
    filepath="./checkpoints/checkpoint.pt",
    metadata={
        "epoch": 10,
        "git_info": repro.git_info  # Critical for reproducibility!
    },
    model=model,
    optimizer=optimizer
)
```

### 2. Manually Save RNG State for Resumption

If you need to resume training with perfect reproducibility, you are responsible for getting the RNG states and placing them in the `metadata` dictionary.

```python
# Assuming 'repro' is an instance of ReproducibilityManager
save_checkpoint(
    filepath="./checkpoints/checkpoint.pt",
    metadata={
        "epoch": 10,
        "rng_states": repro.get_rng_states() # Manually add for reproducible resumption
    },
    model=model,
    optimizer=optimizer
)
```

### 3. Automatic Handling for DDP and `torch.compile`

The checkpointer is designed to work seamlessly with common PyTorch wrappers, removing boilerplate and potential errors from your training scripts.

-   **`torch.nn.parallel.DistributedDataParallel` (DDP)**: You no longer need to manually unwrap the model by accessing `.module`. The checkpointer automatically detects a DDP-wrapped model and saves the underlying model's state dictionary.
-   **`torch.compile`**: Similarly, if you pass a compiled model, the checkpointer automatically saves the state dictionary of the original, uncompiled model.

This means you can pass your model object directly to `save_checkpoint` and `load_checkpoint`, regardless of whether it's wrapped in DDP, compiled with `torch.compile`, or both.

**Example:**

```python
# model can be a raw model, a DDP-wrapped model, or a compiled model
model = DDP(torch.compile(MyModel()), device_ids=[local_rank])

# No need to call model.module or access _orig_mod
save_checkpoint(
    filepath="./checkpoints/checkpoint.pt",
    model=model, # Pass the wrapped model directly
    optimizer=optimizer
)

# To load, you can also pass the wrapped model directly
# Tensors will be automatically moved to the correct device
load_checkpoint(
    filepath="./checkpoints/checkpoint.pt",
    model=model,
    optimizer=optimizer
)
```

## Testing

Comprehensive tests are in `tests/src/gpt_lab/test_checkpointer.py`:

```bash
pytest tests/src/gpt_lab/test_checkpointer.py -v
```

**Test coverage includes:**
- Model and optimizer state preservation
- Metadata saving/loading
- Compatibility with `torch.compile` and DDP
- Automatic device mapping