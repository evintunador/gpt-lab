# nn.Modules Catalog

Neural network components organized by functionality.

## Overview

The nn_modules catalog provides PyTorch `nn.Module` subclasses for building models. Components are automatically discovered and tested across all active roots.

**Namespace**: `gpt_lab.nn_modules`

## Categories

Modules are organized into categories:

- **sequence_mixing**: Attention mechanisms
- **channel_mixing**: Feed-forward and MLP layers
- **norms**: Normalization layers
- **activations**: Activation functions

See `packs-core.md` for available core modules.
Other packs and specific experiments may create more categories and/or sub-categories.

## Using Modules

```python
from gpt_lab.nn_modules.sequence_mixing import CausalSelfAttention
from gpt_lab.nn_modules.channel_mixing import GatedMLP
from gpt_lab.nn_modules.norms import RMSNorm

class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.attention = CausalSelfAttention(dim, n_heads)
        self.mlp = GatedMLP(dim)
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
    
    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
```

## Testing

Modules are automatically tested if they define `__test_config__`:

```python
class MyModule(nn.Module):
    __test_config__ = {
        "constructor_args": {
            "dim": 64,
            "n_heads": 4
        },
        "forward_args": {
            "x": {"shape": (2, 10, 64), "dtype": "float32"}
        },
        "expected_output_shape": (2, 10, 64)
    }
    
    def __init__(self, dim, n_heads):
        super().__init__()
        # Implementation
    
    def forward(self, x):
        # Implementation
        return output
```

Run tests:

```bash
pytest src/gpt_lab/nn_modules/tests/ -v
```

**Test features:**
- Validates forward pass outputs
- Checks gradient computation
- Verifies numerical stability
- Generates failure heatmaps in `test_artifacts/nn_modules/`

## Benchmarking

Benchmark module performance:

```bash
python -m gpt_lab.nn_modules.catalog_benchmark \
    --modules MyModule OtherModule \
    --output benchmark_results.csv

# View results
marimo edit notebooks/nn_modules_bench.py
```

## Contributing

Add modules to:
- Core: `catalogs/core/gpt_lab/nn_modules/`
- Pack: `catalogs/packs/<pack>/gpt_lab/nn_modules/`
- Experiment: `experiments/<exp>/gpt_lab/nn_modules/`

Include `__test_config__` for automated testing.
