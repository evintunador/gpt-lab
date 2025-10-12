# Smart Training Loop

The smart training loop API (`smart_train()`) automatically composes atomic training features based on your arguments, enabling rapid experimentation without manual loop construction.

## Overview

`smart_train()` analyzes your kwargs, selects appropriate atomic features, and either:
1. Executes a single feature directly if only one is needed
2. Uses an LLM to compile an optimized combined training loop if multiple features are needed

The compiled loops are cached for reuse.

## Usage

```python
from gpt_lab.train_loops import smart_train

result = smart_train(
    model=model,
    optimizer=optimizer,
    train_loader=train_loader,
    # Atomic features are automatically selected from these kwargs
    num_epochs=10,
    grad_accum_steps=4,
    mixed_precision=True,
    grad_clip=1.0,
    lr_schedule="cosine",
    val_loader=val_loader,
    validate_every_n_epochs=1
)
```

**Returns**: Dictionary with at least `{"model": model}` plus any feature-specific results.

## How It Works

### 1. Feature Discovery

`smart_train()` discovers all atomic features across active roots:

```python
# Searches these locations:
# - catalogs/core/gpt_lab/train_loops/
# - catalogs/packs/<pack>/gpt_lab/train_loops/
# - experiments/<exp>/gpt_lab/train_loops/
```

### 2. Feature Selection

Based on your kwargs, selects the most specific features:

```python
# Example: User provides val_loader and patience
smart_train(
    model=model,
    optimizer=optimizer,
    train_loader=train_loader,
    val_loader=val_loader,
    patience=5
)
# Selects: early_stopping (not just validation, since early_stopping is more specific)
```

### 3. Loop Compilation

**Single feature**: Executes directly
**Multiple features**: Uses LLM to generate optimized combined loop

The LLM:
- Reads source code of selected atomic features
- Combines them efficiently
- Validates the generated code
- Caches the result in `artifacts/train_loops/llm_compiled/`

### 4. Execution

Runs the compiled (or direct) training loop with your arguments.

## Available Features

See `catalogs-train-loops.md` for complete list. Key features include:

**Optimization**:
- `grad_accum_steps` - Gradient accumulation
- `mixed_precision` - Automatic mixed precision
- `grad_clip` - Gradient clipping
- `lr_schedule` - Learning rate scheduling

**Validation & Checkpointing**:
- `val_loader` + `validate_every_n_epochs` - Validation
- `patience` + `min_delta` - Early stopping
- `checkpoint_every_n_epochs` - Periodic checkpointing
- `save_best_model` - Save best model

**Monitoring**:
- `use_tqdm` - Progress bars
- `log_every_n_steps` - Logging frequency

## Advanced Configuration

### Custom LLM Compiler

```python
result = smart_train(
    model=model,
    optimizer=optimizer,
    train_loader=train_loader,
    llm_compiler_model="openai/gpt-4",  # Default: "anthropic/claude-sonnet-4"
    api_key="your-api-key",  # Optional, reads from .env by default
    # ... other kwargs
)
```

### Viewing Compiled Loops

Compiled loops are saved in `artifacts/train_loops/llm_compiled/`:

```bash
ls catalogs/core/gpt_lab/artifacts/train_loops/llm_compiled/
# Shows: grad_accum-mixed_precision.py, etc.
```

You can read these to understand how features were combined.

## Custom Atomic Features

Add features in your experiment:

```python
# experiments/my_exp/gpt_lab/train_loops/my_feature.py

def run_training(model, optimizer, train_loader, my_param=None, **kwargs):
    """
    Custom training feature.
    
    Args:
        my_param: My custom parameter
    """
    # Implementation
    for batch in train_loader:
        # Training logic
        pass
    
    return {"model": model, "my_result": some_value}
```

Then use it:

```python
smart_train(
    model=model,
    optimizer=optimizer,
    train_loader=train_loader,
    my_param=42  # Automatically discovered and included
)
```

## Testing

Smart train is tested across various feature combinations:

```bash
pytest src/gpt_lab/train_loops/tests/test_smart_api.py -v
```

## Best Practices

### 1. Start Simple

Begin with basic features and add complexity:

```python
# Start
smart_train(model, optimizer, train_loader, num_epochs=10)

# Then add
smart_train(model, optimizer, train_loader, num_epochs=10, grad_accum_steps=4)

# Then add more
smart_train(
    model, optimizer, train_loader, val_loader,
    num_epochs=10, grad_accum_steps=4, validate_every_n_epochs=1
)
```

### 2. Leverage Compiled Loops

Once a loop is compiled for a feature combination, it's cached and reused instantly.

### 3. Inspect Generated Code

Review compiled loops to understand how features interact:

## Contributing

To add atomic features:
1. Create `.py` file in `train_loops/` directory
2. Define `run_training(model, optimizer, train_loader, **kwargs)` function
3. Document kwargs in function signature and docstring
4. Return dictionary with at least `{"model": model}`

See existing features in `catalogs/core/gpt_lab/train_loops/` for examples.

## Related Documentation

- `catalogs-train-loops.md` - Complete list of atomic features
- `components-configuration.md` - Parameter configuration
- `packs-core.md` - Core training loop features
