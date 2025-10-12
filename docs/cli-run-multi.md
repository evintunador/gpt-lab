# Run Multi CLI

The `run_multi.py` script executes multiple experiment runs with different parameter configurations, enabling systematic hyperparameter sweeps and ablation studies.

## Overview

This CLI tool:
- Runs multiple experiments with different configurations
- Supports grid search, list, and custom parameter variations
- Executes runs in parallel or sequential mode
- Tracks success/failure of each run
- Provides detailed logging and summaries

## Usage

```bash
# Basic usage
python CLIs/run_multi.py --config path/to/multi_run.yaml

# Override execution mode
python CLIs/run_multi.py --config multi_run.yaml --execution.mode sequential

# Override max parallel workers
python CLIs/run_multi.py --config multi_run.yaml --execution.max_workers 4

# Override other parameters
python CLIs/run_multi.py --config multi_run.yaml --parameters.learning_rate "[0.001, 0.0001]"
```

## Configuration File

Create a `multi_run.yaml` file:

```yaml
# Name for this multi-run batch
name: "learning_rate_sweep"

# Command to execute for each run
command: "python experiments/nano_gpt/main.py"

# Parameter variations
parameters:
  learning_rate: [0.001, 0.0005, 0.0001]
  batch_size: [32, 64]
  model.n_layers: [6, 12]

# Execution settings (optional)
execution:
  mode: "parallel"  # or "sequential"
  max_workers: 4    # for parallel mode
  continue_on_error: true  # keep running if one fails
```

## Parameter Types

### Grid Search

All combinations of parameters:

```yaml
parameters:
  learning_rate: [0.001, 0.0001]
  batch_size: [32, 64]
# Generates 4 runs: (0.001, 32), (0.001, 64), (0.0001, 32), (0.0001, 64)
```

### List Parameters

Zip parameters together:

```yaml
parameters:
  __mode__: "zip"
  learning_rate: [0.001, 0.0005, 0.0001]
  weight_decay: [0.1, 0.01, 0.001]
# Generates 3 runs: (0.001, 0.1), (0.0005, 0.01), (0.0001, 0.001)
```

### Mixed Modes

Combine grid and list:

```yaml
parameters:
  learning_rate: [0.001, 0.0001]
  model:
    __mode__: "zip"
    n_layers: [6, 12]
    n_heads: [6, 12]
# Generates: lr × (layers, heads) pairs
```

## Output

The script provides a summary after completion:

```
============================================================
Multi-run 'learning_rate_sweep' completed
Total runs: 12
Successful: 11
Failed: 1
============================================================
```

Each run creates its own output directory:
```
runs/
├── 2025-10-12_14-30-45_a1b2c3d/  # lr=0.001, bs=32
├── 2025-10-12_14-31-10_a1b2c3d/  # lr=0.001, bs=64
├── 2025-10-12_14-31-35_a1b2c3d/  # lr=0.0001, bs=32
└── ...
```

## Integration with Configuration System

The multi-runner uses `get_config()` from `gpt_lab.configuration`, so you can override any parameter:

```bash
# Override command
python CLIs/run_multi.py --config sweep.yaml --command "python experiments/other/main.py"

# Override execution mode
python CLIs/run_multi.py --config sweep.yaml --execution.mode sequential

# Override parameters
python CLIs/run_multi.py --config sweep.yaml --parameters.learning_rate "[0.001]"
```

## Best Practices

### 1. Start with Sequential for Debugging

```yaml
execution:
  mode: "sequential"
  # Switch to parallel once everything works
```

### 2. Use Continue on Error

```yaml
execution:
  continue_on_error: true  # Don't stop entire sweep if one run fails
```

### 3. Name Your Sweeps

```yaml
name: "lr_sweep_2025_10_12"  # Include date for organization
```

### 4. Test Small First

```yaml
# Test with 2-3 runs first
parameters:
  learning_rate: [0.001, 0.0001]  # Just 2 values

# Then expand
# learning_rate: [0.001, 0.0005, 0.0001, 0.00005, 0.00001]
```

### 5. Save Multi-Run Configs

```bash
experiments/nano_gpt/
├── config.yaml           # Base config
├── lr_sweep.yaml        # Multi-run config
├── arch_ablation.yaml   # Another sweep
└── main.py
```

## Troubleshooting

### Runs Fail to Start

Check command syntax:
```yaml
command: "python experiments/nano_gpt/main.py"  # Include full path
```

### Out of Memory with Parallel

Reduce workers:
```yaml
execution:
  max_workers: 2  # Or switch to sequential
```

### Parameters Not Applied

Ensure parameter names match your config structure:
```yaml
# If your config.yaml has:
# training:
#   learning_rate: 0.001

# Use:
parameters:
  training.learning_rate: [0.001, 0.0001]
```

## Advanced Usage

### Conditional Parameters

```yaml
parameters:
  model.type: ["transformer", "lstm"]
  model.transformer_layers: [6, 12]  # Only used if type=transformer
  model.lstm_layers: [2, 4]          # Only used if type=lstm
```

### Custom Parameter Generators

```yaml
# In your multi_run.yaml, reference a generator script
parameter_generator: "scripts/generate_params.py"
```

```python
# scripts/generate_params.py
def generate_params():
    return [
        {"lr": 0.001 * (0.5 ** i), "wd": 0.1 * (0.5 ** i)}
        for i in range(10)
    ]
```

