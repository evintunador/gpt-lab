# Testing

GPT-Lab uses **discovery-based testing** where tests automatically find catalog items across active roots. This ensures all components work correctly in their intended contexts.

## Overview

Tests are located in `src/gpt_lab/**/tests/` and discover catalog items from active roots. What gets tested depends on which experiments and packs are activated.

**Key insight**: The same test code runs against different sets of catalog items based on activation configuration.

## Configuration

### pyproject.toml

```toml
[tool.pytest.ini_options]
testpaths = ["src/gpt_lab"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
```

Tests are collected from `src/gpt_lab/` but discover catalog items from active roots.

## What Runs By Default

**From repo root** (no activation):
```bash
pytest
# Tests only core catalog items
```

**From experiment directory**:
```bash
cd experiments/nano_gpt
pytest
# Tests: nano_gpt + its includes + packs + core
```

**With environment variables**:
```bash
export GPT_LAB_CURRENT_EXPERIMENT=nano_gpt
export GPT_LAB_ACTIVE_PACKS=nlp
pytest
# Tests: nano_gpt + nlp pack + core
```

## Discovery Patterns

Each catalog type has specific discovery patterns:

### nn_modules

**Discovery**:
- Walks `gpt_lab.nn_modules` namespace
- Finds classes with `__test_config__` attribute
- Generates parameterized tests

**Tests**:
- Forward pass shape validation
- Gradient computation
- Numerical stability
- Cross-implementation comparison

**Example**:
```python
# Your module
class MyAttention(nn.Module):
    __test_config__ = {
        "constructor_args": {"dim": 64, "n_heads": 4},
        "forward_args": {"x": {"shape": (2, 10, 64), "dtype": "float32"}},
        "expected_output_shape": (2, 10, 64)
    }
```

**Run tests**:
```bash
pytest src/gpt_lab/nn_modules/tests/ -v
```

### optimizers

**Discovery**:
- Walks `gpt_lab.optimizers` namespace
- Finds `torch.optim.Optimizer` subclasses

**Tests**:
- Universal learning test (trains small model, asserts loss reduction)
- Gradient update verification
- Numerical stability

**Run tests**:
```bash
pytest src/gpt_lab/optimizers/tests/ -v
```

### train_loops

**Discovery**:
- Finds `run_training` functions in `gpt_lab.train_loops`
- Tests atomic features individually and in combination

**Tests**:
- Feature composition
- Parameter passing
- Return value validation
- Integration with smart_train

**Run tests**:
```bash
pytest src/gpt_lab/train_loops/tests/ -v
```

### benchmarks, data_sources, models

Tests are catalog-specific and test functionality appropriate to each type.

## Common Testing Scenarios

### Test Core Only

```bash
# Clear all activation
unset GPT_LAB_CURRENT_EXPERIMENT
unset GPT_LAB_ACTIVE_EXPERIMENTS
unset GPT_LAB_ACTIVE_PACKS

# Or explicitly set to none
export GPT_LAB_CURRENT_EXPERIMENT=none
export GPT_LAB_ACTIVE_EXPERIMENTS=
export GPT_LAB_ACTIVE_PACKS=

pytest
```

### Test Specific Experiment

```bash
# Method 1: Use directory
cd experiments/nano_gpt
pytest

# Method 2: Use environment variables
export GPT_LAB_CURRENT_EXPERIMENT=nano_gpt
export GPT_LAB_ACTIVE_EXPERIMENTS=nano_gpt
pytest
```

### Test Experiment + Pack

```bash
export GPT_LAB_CURRENT_EXPERIMENT=my_nlp_exp
export GPT_LAB_ACTIVE_EXPERIMENTS=my_nlp_exp
export GPT_LAB_ACTIVE_PACKS=nlp
pytest
```

### Test Pack Only (No Experiment)

```bash
export GPT_LAB_CURRENT_EXPERIMENT=none
export GPT_LAB_ACTIVE_EXPERIMENTS=
export GPT_LAB_ACTIVE_PACKS=nlp
pytest
```

### Test All Experiments in Isolation

Use the CLI tool to run each experiment's tests separately:

```bash
python CLIs/pytest_all_experiments.py

# With pytest arguments
python CLIs/pytest_all_experiments.py --pytest-args -v

# Test specific experiments
python CLIs/pytest_all_experiments.py --include nano_gpt modded_nano_gpt

# Exclude experiments
python CLIs/pytest_all_experiments.py --exclude broken_exp
```

See [CLI: Pytest All Experiments](cli-pytest-all-experiments.md) for details.

## Test Output and Artifacts

### Test Artifacts

Some tests generate artifacts:

```
test_artifacts/
├── nn_modules/
│   ├── test_mymodule_forward/
│   │   ├── heatmap.png
│   │   └── diff_values.npy
│   └── test_mymodule_gradients/
│       └── gradient_comparison.png
└── optimizers/
    └── test_adamw_convergence/
        └── loss_curve.png
```

### Verbose Output

```bash
# See discovered items
pytest -v

# See detailed failures
pytest -vv

# Show print statements
pytest -s
```

### Debugging

```bash
# Drop into debugger on failure
pytest --pdb

# Show local variables on failure
pytest -l

# Run specific test
pytest src/gpt_lab/nn_modules/tests/test_catalog.py::test_forward_pass -v
```

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Test

on: [push, pull_request]

jobs:
  test-core:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - run: pip install -e '.[dev]'
      
      # Test core only
      - name: Test core components
        run: |
          export GPT_LAB_CURRENT_EXPERIMENT=none
          export GPT_LAB_ACTIVE_EXPERIMENTS=
          export GPT_LAB_ACTIVE_PACKS=
          pytest --cov=src/gpt_lab --cov-report=xml
      
      - name: Upload coverage
        uses: codecov/codecov-action@v3

  test-experiments:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - run: pip install -e '.[dev]'
      
      # Test all experiments
      - name: Test all experiments
        run: python CLIs/pytest_all_experiments.py --pytest-args -v

  test-packs:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        pack: [nlp, cv]
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - run: pip install -e '.[${{ matrix.pack }}]'
      
      - name: Test pack
        run: |
          export GPT_LAB_CURRENT_EXPERIMENT=none
          export GPT_LAB_ACTIVE_PACKS=${{ matrix.pack }}
          pytest -v
```

## Writing Tests for Catalog Items

### nn_modules

Add `__test_config__` to your module:

```python
class MyModule(nn.Module):
    __test_config__ = {
        "constructor_args": {
            "dim": 64,
            "n_heads": 4,
            "dropout": 0.1
        },
        "forward_args": {
            "x": {"shape": (2, 10, 64), "dtype": "float32"}
        },
        "expected_output_shape": (2, 10, 64),
        "tolerances": {
            "forward": 1e-5,
            "gradient": 1e-4
        }
    }
    
    def __init__(self, dim, n_heads, dropout=0.0):
        super().__init__()
        # Implementation
    
    def forward(self, x):
        # Implementation
        return output
```

### optimizers

Simply subclass `torch.optim.Optimizer`:

```python
class MyOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        defaults = dict(lr=lr)
        super().__init__(params, defaults)
    
    def step(self, closure=None):
        # Implementation
        pass
```

Tests will automatically discover and verify it reduces loss.

### train_loops

Follow the standard signature:

```python
def run_training(model, optimizer, train_loader, my_param=None, **kwargs):
    """My training loop feature."""
    # Implementation
    return {"model": model, "my_metric": value}
```

## Troubleshooting

### No Tests Discovered

**Problem**: `collected 0 items`

**Solutions**:

1. **Wrong directory**:
   ```bash
   # From subdirectory, specify testpath
   pytest src/gpt_lab/
   ```

2. **No activation**:
   ```bash
   # Check what's active
   python CLIs/print_active_paths.py -v
   
   # Activate something
   export GPT_LAB_ACTIVE_PACKS=nlp
   pytest
   ```

### Tests Fail Unexpectedly

**Problem**: Tests pass in isolation but fail when combined.

**Solution**: Check for conflicts in active roots:
```bash
python CLIs/print_active_paths.py -v
# Look for multiple implementations of same component
```

### Import Errors During Tests

**Problem**: `ImportError: cannot import name 'X'`

**Solution**: Verify component is in an active root:
```bash
python CLIs/print_active_paths.py -v
# Check if root containing X is listed
```

### Test Artifacts Not Generated

**Problem**: Failure heatmaps or artifacts missing.

**Solution**: Check test configuration and permissions:
```bash
ls -la test_artifacts/
# Ensure directory is writable
```

## Best Practices

### 1. Test in Multiple Contexts

Test your components with different activations:

```bash
# Test alone
pytest

# Test with pack
export GPT_LAB_ACTIVE_PACKS=nlp
pytest

# Test with other experiments
export GPT_LAB_ACTIVE_EXPERIMENTS=my_exp,other_exp
pytest
```

### 2. Use Parameterized Tests

When writing custom tests, use pytest parametrization:

```python
import pytest

@pytest.mark.parametrize("dim,n_heads", [
    (64, 4),
    (128, 8),
    (256, 16)
])
def test_attention_shapes(dim, n_heads):
    attention = MyAttention(dim, n_heads)
    x = torch.randn(2, 10, dim)
    output = attention(x)
    assert output.shape == (2, 10, dim)
```

### 3. Clean Up Test Artifacts

```bash
# Remove old artifacts
rm -rf test_artifacts/

# Run tests fresh
pytest
```

### 4. Use Markers for Slow Tests

```python
@pytest.mark.slow
def test_full_training_convergence():
    # Long-running test
    pass
```

```bash
# Skip slow tests
pytest -m "not slow"

# Run only slow tests
pytest -m slow
```

### 5. Verify Before Committing

```bash
# Run full test suite
pytest

# Test all experiments
python CLIs/pytest_all_experiments.py

# Check coverage
pytest --cov=src/gpt_lab --cov-report=html
open htmlcov/index.html
```

## Related Documentation

- [Architecture](architecture.md) - How discovery works across roots
- [Configuration](config.md) - How activation affects tests
- [CLI: Pytest All Experiments](cli-pytest-all-experiments.md) - Testing all experiments
- Individual catalog documentation for testing specific types
