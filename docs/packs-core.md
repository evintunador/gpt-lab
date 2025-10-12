# Core Pack

The core pack (`catalogs/core/gpt_lab/`) provides fundamental building blocks that are domain-agnostic and useful across different types of ML experiments.

## Overview

The core pack includes:
- **nn_modules**: Essential neural network components (attention, MLPs, norms, activations)
- **optimizers**: Training optimizers (AdamW, Muon)
- **train_loops**: Atomic training loop features for composable training logic

These components are always available and provide the foundation for all experiments.

## Usage

### Building Models with Core Components

```python
import torch.nn as nn
# No activation needed - core is always active
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

### Using Core Train Loop Features

```python
from gpt_lab.train_loops import smart_train

# Compose training loop from atomic features
smart_train(
    model=model,
    train_loader=train_loader,
    optimizer=optimizer,
    # Atomic features
    num_epochs=10,
    grad_accum_steps=4,
    mixed_precision=True,
    grad_clip=1.0,
    lr_schedule="cosine",
    checkpoint_every_n_epochs=1,
    validate_every_n_epochs=1,
    val_loader=val_loader
)
```

## Testing

Core pack items are tested as part of the main test suite:

```bash
# Test with core pack only
pytest src/gpt_lab/tests/ -v

# Test specific catalog type
pytest src/gpt_lab/nn_modules/tests/ -v
pytest src/gpt_lab/optimizers/tests/ -v
pytest src/gpt_lab/train_loops/tests/ -v
```

## File Structure

```
catalogs/core/gpt_lab/
├── nn_modules/
│   ├── activations/
│   │   ├── __init__.py
│   │   └── relu2.py
│   │   └── ...
│   ├── channel_mixing/
│   │   ├── __init__.py
│   │   ├── fp8_linear.py
│   │   ├── gated_mlp.py
│   │   └── mlp.py
│   │   └── ...
│   ├── norms/
│   │   ├── __init__.py
│   │   └── rms_norm.py
│   │   └── ...
│   └── sequence_mixing/
│       ├── __init__.py
│       ├── causal_self_attention.py
│       └── flex_self_attention.py
│   │   └── ...
├── optimizers/
│   ├── __init__.py
│   ├── adamw.py
│   ├── muon.py
│   └── ...
├── train_loops/
│   ├── __init__.py
│   ├── grad_accum.py
│   ├── mixed_precision.py
│   ├── lr_scheduling.py
│   └── ...
└── artifacts/
    └── train_loops/
        └── llm_compiled/
    └── nn_modules/
        └── bench_results/
    └── optimizers/
        └── bench_results/
```

