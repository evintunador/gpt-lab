# Checkpointer

The `gpt_lab.checkpointer` module provides flexible checkpoint saving and loading for reproducible ML experiments.

## Overview

The checkpointer component is designed to:
- Save and load model, optimizer, and any stateful PyTorch objects
- Preserve RNG states for perfect reproducibility
- Store arbitrary metadata (metrics, epoch, step, git info, etc.)
- Work seamlessly with distributed training
- Provide simple, flexible APIs that work with any PyTorch object with `state_dict()`

## Core Functions

### `save_checkpoint()`

Saves a comprehensive training checkpoint:

```python
from gpt_lab.checkpointer import save_checkpoint

filepath = save_checkpoint(
    save_dir="./checkpoints",
    filename="epoch_10.pt",
    metadata={
        "epoch": 10,
        "step": 5000,
        "best_val_loss": 0.342,
        "git_info": repro_manager.git_info
    },
    save_rng_state=True,
    model=model,
    optimizer=optimizer,
    scheduler=lr_scheduler
)
# Returns: "./checkpoints/epoch_10.pt"
```

**Parameters:**
- `save_dir`: Directory to save checkpoint (created if doesn't exist)
- `filename`: Name of checkpoint file
- `metadata`: Dictionary of arbitrary metadata to save
- `save_rng_state`: If `True`, saves torch, numpy, and random RNG states
- `**stateful_objects`: Keyword arguments for any objects with `state_dict()` method

**Returns:** Full path to the saved checkpoint file

**Checkpoint Structure:**
```python
{
    'metadata': {...},          # Your custom metadata
    'rng_states': {             # Optional, if save_rng_state=True
        'torch': <torch_state>,
        'numpy': <numpy_state>,
        'random': <random_state>
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

returned_data = load_checkpoint(
    filepath="./checkpoints/epoch_10.pt",
    map_location="cuda:0",
    model=model,
    optimizer=optimizer,
    scheduler=lr_scheduler
)

# Returned data contains metadata and RNG states
epoch = returned_data['metadata']['epoch']
step = returned_data['metadata']['step']

# Optionally restore RNG states
if 'rng_states' in returned_data:
    torch.set_rng_state(returned_data['rng_states']['torch'])
    np.random.set_state(returned_data['rng_states']['numpy'])
    random.setstate(returned_data['rng_states']['random'])
```

**Parameters:**
- `filepath`: Path to checkpoint file
- `map_location`: Device to load tensors to (`'cpu'`, `'cuda:0'`, etc.)
- `**stateful_objects`: Objects to load state into (must match keys from save)

**Returns:** Dictionary containing all non-state-dict data (metadata, RNG states)

**Behavior:**
- Calls `load_state_dict()` on each provided object in-place
- Returns metadata and RNG states for manual handling
- Prints warnings for missing keys or objects

## Best Practices

### 1. Always Include Git Info

Use `ReproducibilityManager` to capture git information:

```python
from gpt_lab.reproducibility import ReproducibilityManager

repro = ReproducibilityManager()

save_checkpoint(
    save_dir="./checkpoints",
    filename="checkpoint.pt",
    metadata={
        "epoch": 10,
        "git_info": repro.git_info  # Critical for reproducibility!
    },
    model=model,
    optimizer=optimizer
)
```

### 2. Save RNG State for Resumption

Always set `save_rng_state=True` if you might need to resume training:

```python
save_checkpoint(
    save_dir="./checkpoints",
    filename="checkpoint.pt",
    metadata={"epoch": 10},
    save_rng_state=True,  # Ensures reproducible resumption
    model=model,
    optimizer=optimizer
)
```

### 3. Unwrap DDP Models

When using `DistributedDataParallel`, save the underlying module:

```python
from torch.nn.parallel import DistributedDataParallel as DDP

model = DDP(MyModel(), device_ids=[local_rank])

# Save the underlying module, not the DDP wrapper
save_checkpoint(
    save_dir="./checkpoints",
    filename="checkpoint.pt",
    model=model.module,  # Not model!
    optimizer=optimizer
)
```

## Testing

Comprehensive tests are in `src/gpt_lab/tests/test_checkpointer.py`:

```bash
pytest src/gpt_lab/tests/test_checkpointer.py -v
```

**Test coverage includes:**
- Model and optimizer state preservation
- Metadata and RNG state saving/loading
- Git info integration
- Cross-device loading (save on CUDA, load on CPU)
- Optimizer state with momentum buffers

## Contributing

To contribute to the checkpointer:

1. **Adding Features**: Extend functionality in `src/gpt_lab/checkpointer.py`
   - Add checkpoint validation functions
   - Add checkpoint conversion utilities (e.g., for format migration)
   - Add checkpoint merging/averaging utilities
   - Add automatic checkpoint cleanup based on metrics

2. **Adding Tests**: Add test cases to `src/gpt_lab/tests/test_checkpointer.py`
   - Test edge cases (empty models, zero-sized tensors)
   - Test with different optimizer types
   - Test checkpoint migration between versions
   - Test checkpoint corruption handling

3. **Guidelines**:
   - Maintain backward compatibility with existing checkpoint format
   - Keep the API simple and flexible
   - Always test on both CPU and CUDA (if available)
   - Document any changes to checkpoint structure
