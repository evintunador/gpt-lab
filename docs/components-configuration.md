# Configuration

The `gpt_lab.configuration` module provides flexible configuration management that merges YAML files with command-line arguments, supporting nested configuration via dot-notation.

## Overview

The configuration component is designed to:
- Load base configuration from YAML files
- Override config values via command-line arguments
- Support nested configuration with dot-notation (e.g., `--model.dim=512`)
- Automatically discover `config.yaml` in the script's directory
- Type inference for CLI arguments (int, float, bool, string)
- Recursive merging of nested dictionaries

## Key Function

### `get_config()`

Builds a configuration dictionary by merging YAML and CLI arguments:

```python
import argparse
from gpt_lab.configuration import get_config

parser = argparse.ArgumentParser()
# Optionally add your own arguments
parser.add_argument('--verbose', action='store_true')

# Load configuration
config = get_config(parser)

# Access nested config
learning_rate = config['training']['learning_rate']
model_dim = config['model']['dim']
```

**Parameters:**
- `parser`: An `argparse.ArgumentParser` instance

**Returns:**
- Dictionary containing merged configuration

**Behavior:**
1. Automatically adds `--config` argument if not present
2. Loads YAML from config file (defaults to `./config.yaml` if exists)
3. Parses CLI arguments (both registered and unknown)
4. Merges CLI args into YAML config (CLI takes precedence)
5. Returns final merged dictionary

## Configuration Precedence

Configuration values are merged in this order (later overrides earlier):

1. **YAML file** - Base configuration
2. **Registered parser arguments** - Arguments added via `parser.add_argument()`
3. **Unknown CLI arguments** - Dot-notation arguments (e.g., `--model.dim=512`)

## YAML Configuration

Create a `config.yaml` file in your experiment directory:

```yaml
# config.yaml
seed: 42

model:
  dim: 512
  n_layers: 6
  n_heads: 8
  dropout: 0.1

training:
  batch_size: 32
  learning_rate: 0.001
  num_epochs: 100
  gradient_accumulation_steps: 1

data:
  dataset: "wikitext"
  max_length: 1024
  num_workers: 4
```

## Usage Examples

### Basic Configuration Loading

```python
import argparse
from gpt_lab.configuration import get_config

def main():
    parser = argparse.ArgumentParser(description="Train model")
    config = get_config(parser)
    
    print(f"Model dim: {config['model']['dim']}")
    print(f"Learning rate: {config['training']['learning_rate']}")
    
    # Use configuration
    model = MyModel(
        dim=config['model']['dim'],
        n_layers=config['model']['n_layers']
    )

if __name__ == "__main__":
    main()
```

Run with:
```bash
python train.py  # Uses config.yaml
```

### Overriding with CLI Arguments

Override nested configuration values using dot-notation:

```bash
# Override single value
python train.py --training.learning_rate=0.0001

# Override multiple values
python train.py \
    --model.dim=1024 \
    --model.n_layers=12 \
    --training.batch_size=64

# Mix with registered arguments
python train.py --verbose --training.learning_rate=0.0001
```

In code:

```python
parser = argparse.ArgumentParser()
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--debug', action='store_true')

config = get_config(parser)

# Registered arguments are in config
if config.get('verbose'):
    print("Verbose mode enabled")

# Dot-notation overrides work
lr = config['training']['learning_rate']  # Uses CLI value if provided
```

### Using Custom Config File

Specify a different config file:

```bash
python train.py --config=experiments/my_exp/custom_config.yaml
```

```python
parser = argparse.ArgumentParser()
# You can also set default
parser.add_argument('--config', default='path/to/config.yaml')

config = get_config(parser)
```

### Type Inference

CLI arguments are automatically converted to appropriate types:

```bash
# Integer
python train.py --model.dim=512  # config['model']['dim'] = 512 (int)

# Float
python train.py --training.learning_rate=0.001  # 0.001 (float)

# Boolean
python train.py --training.use_amp=true  # True (bool)
python train.py --training.use_amp=false  # False (bool)

# String
python train.py --data.dataset=wikitext  # "wikitext" (str)
```

**Boolean string recognition:**
- True: `'yes'`, `'true'`, `'on'`, `'1'`, `'t'` (case-insensitive)
- False: `'no'`, `'false'`, `'off'`, `'0'`, `'f'` (case-insensitive)

## Best Practices

### 1. Use Config Validation

Always validate your configuration before starting expensive training:

```python
def validate_config(config):
    assert config['model']['dim'] % config['model']['n_heads'] == 0, \
        "model.dim must be divisible by model.n_heads"
    
    assert config['training']['learning_rate'] > 0, \
        "learning_rate must be positive"
    
    assert config['training']['batch_size'] > 0, \
        "batch_size must be positive"
```

### 2. Create Config Variants

```bash
# Base config
python train.py --config=config.yaml

# Large model variant
python train.py --config=config_large.yaml

# Debug variant (small, fast)
python train.py --config=config_debug.yaml
```

### 3. Save Final Config with Results

Always save the final merged configuration alongside your results:

```python
with ReproducibilityManager(output_dir="./runs") as repro:
    config = get_config(parser)
    
    # Save configuration
    with open(f"{repro.output_dir}/config.yaml", 'w') as f:
        yaml.dump(config, f)
    
    train(config)
```

## Contributing

To contribute to the configuration component:

1. **Adding Features**: Extend functionality in `src/gpt_lab/configuration.py`
   - Add JSON config support
   - Add config validation utilities
   - Add config diffing utilities
   - Add environment variable substitution in YAML

2. **Adding Tests**: Add test cases to `src/gpt_lab/tests/test_configuration.py`
   - Test type inference edge cases
   - Test deeply nested config merging
   - Test various YAML structures
   - Test error handling

3. **Guidelines**:
   - Maintain backward compatibility
   - Handle edge cases gracefully
   - Preserve type information
   - Document any breaking changes