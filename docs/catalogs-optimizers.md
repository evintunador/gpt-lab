# Optimizers Catalog

Training optimizers for different optimization strategies.

## Overview

The optimizers catalog provides `torch.optim.Optimizer` subclasses. Optimizers are automatically discovered and tested across all active roots.

**Namespace**: `gpt_lab.optimizers`

## Available Optimizers

See `packs-core.md` for core optimizers (AdamW, Muon).

## Using Optimizers

```python
from gpt_lab.optimizers import AdamW, Muon

# AdamW (standard)
optimizer = AdamW(
    model.parameters(),
    lr=1e-3,
    weight_decay=0.1,
    betas=(0.9, 0.999)
)

# Muon (optimized for transformers)
optimizer = Muon(
    model.parameters(),
    lr=2e-3
)
```

## Testing

Optimizers are automatically tested:

```bash
pytest src/gpt_lab/optimizers/tests/ -v
```

**Test methodology:**
- Trains a small model for several epochs
- Asserts loss reduction
- Verifies gradient updates
- Checks numerical stability

## Benchmarking

Compare optimizer performance:

```bash
python -m gpt_lab.optimizers.catalog_benchmark \
    --optimizers AdamW Muon \
    --learning-rates 0.001,0.0001 \
    --output optimizer_comparison.csv

# View results
marimo edit notebooks/optimizers_bench.py
```

## Contributing

Add optimizers to:
- Core: `catalogs/core/gpt_lab/optimizers/`
- Experiment: `experiments/<exp>/gpt_lab/optimizers/`

Subclass `torch.optim.Optimizer`:

```python
import torch

class MyOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        defaults = dict(lr=lr)
        super().__init__(params, defaults)
    
    def step(self, closure=None):
        # Implementation
        pass
```
