# Architecture

GPT-Lab uses **namespace bootstrapping** to create a unified, composable catalog system where experiments, packs, and core components layer together seamlessly.

## Core Concept: Namespace Bootstrapping

Instead of traditional inheritance or configuration, GPT-Lab manipulates Python's module system at import time to create unified namespaces like `gpt_lab.nn_modules` that combine code from multiple sources.

### How It Works

When you import `gpt_lab`, the bootstrap process:

1. Discovers all active roots (experiments, packs, core)
2. For each catalog type (`nn_modules`, `optimizers`, etc.), appends each root's directory to the package's `__path__`
3. Python's import system naturally resolves conflicts by precedence order

```python
# After bootstrapping, gpt_lab.nn_modules.__path__ might be:
[
    '/path/to/experiments/my_exp/gpt_lab/nn_modules',      # Highest precedence
    '/path/to/catalogs/packs/nlp/gpt_lab/nn_modules',
    '/path/to/catalogs/core/gpt_lab/nn_modules'            # Lowest precedence
]
```

## Catalog Types

GPT-Lab organizes components into **catalog types**:

| Catalog | Purpose | Example Items |
|---------|---------|---------------|
| `nn_modules` | PyTorch `nn.Module` components | Attention, MLP, RMSNorm |
| `optimizers` | Training optimizers | AdamW, Muon |
| `train_loops` | Atomic training features | Gradient accumulation, checkpointing |
| `benchmarks` | Evaluation frameworks | Multiple choice, fill-in-blank |
| `data_sources` | Dataset loaders | FineWeb, HellaSwag |
| `models` | High-level model wrappers | Tokenizers, inference APIs |

## Roots and Precedence

**Roots** are directories containing `gpt_lab/` subdirectories. Active roots are ordered by precedence:

### Precedence Order (Highest to Lowest)

1. **Current experiment** - The main experiment you're working on
2. **Active experiments** - Additional experiments to include
3. **Active packs** - Domain-specific component collections
4. **Core** - Foundation components (always active)

### Example

```python
# Active roots:
ordered_roots = [
    'experiments/my_transformer/',           # Current experiment
    'experiments/nano_gpt/',                 # Additional experiment
    'catalogs/packs/nlp/',                   # NLP pack
    'catalogs/core/'                         # Core (always last)
]

# When you import:
from gpt_lab.nn_modules.sequence_mixing import CausalSelfAttention

# Python searches in order:
# 1. experiments/my_transformer/gpt_lab/nn_modules/sequence_mixing/causal_self_attention.py
# 2. experiments/nano_gpt/gpt_lab/nn_modules/sequence_mixing/causal_self_attention.py
# 3. catalogs/packs/nlp/gpt_lab/nn_modules/sequence_mixing/causal_self_attention.py
# 4. catalogs/core/gpt_lab/nn_modules/sequence_mixing/causal_self_attention.py
```

## Where Roots Live

### Repository Structure

```
gpt-lab/
├── experiments/               # Experiment roots
│   ├── my_exp/
│   │   ├── gpt_lab/
│   │   │   ├── nn_modules/   # Experiment-specific modules
│   │   │   ├── models/
│   │   │   └── train_loops/
│   │   ├── gpt_lab.yaml      # Declares includes
│   │   └── main.py           # Entry point
│   └── nano_gpt/
│       └── gpt_lab/
│
├── catalogs/
│   ├── packs/                # Pack roots
│   │   ├── nlp/
│   │   │   └── gpt_lab/
│   │   │       ├── benchmarks/
│   │   │       ├── data_sources/
│   │   │       └── models/
│   │   └── cv/
│   │       └── gpt_lab/
│   │
│   └── core/                 # Core root (always active)
│       └── gpt_lab/
│           ├── nn_modules/
│           ├── optimizers/
│           └── train_loops/
│
└── src/gpt_lab/              # Main package source (not a root)
    ├── checkpointer.py       # Importable utilities
    ├── logger.py
    └── train_loops/
        ├── smart_api.py      # Smart train implementation
        └── tests/            # Tests discover items from roots
```

### Key Distinction

- **Roots** (`experiments/`, `catalogs/packs/`, `catalogs/core/`): Contain catalog items
- **Main package** (`src/gpt_lab/`): Contains core infrastructure and utilities

## Design Benefits

### 1. Extensibility

Experiments can override or extend any component without modifying shared code:

```python
# experiments/my_exp/gpt_lab/nn_modules/sequence_mixing/causal_self_attention.py
class CausalSelfAttention(nn.Module):
    """My custom implementation - shadows the core version."""
    pass
```

### 2. Composability

Packs provide reusable domain-specific components:

```yaml
# experiments/my_nlp_exp/gpt_lab.yaml
include_packs: ['nlp']  # Adds all NLP benchmarks, data sources, models
```

Now `gpt_lab.benchmarks`, `gpt_lab.data_sources`, etc. include NLP components.

### 3. Determinism

A single ordered list of roots completely defines:
- What's available to import
- Which version takes precedence  
- What tests discover
- What benchmarks run

### 4. No Code Duplication

Instead of copying code to customize it, experiments extend or override at the namespace level.

### 5. Gradual Refinement

Start with core → add packs for your domain → override/add specifics in your experiment.

## Activation

Roots are activated via:
- Environment variables (highest precedence)
- YAML files (fallback)
- Current working directory inference

See [Configuration](config.md) for details.

## Testing

Tests discover catalog items across all active roots, ensuring:
- Your experiment's items work
- They're compatible with included packs
- Core components remain functional

See [Testing](testing.md) for details.

## Practical Example

### Scenario: Custom Attention for NLP

```python
# Start with core attention
from gpt_lab.nn_modules.sequence_mixing import CausalSelfAttention

# Works, but you want to try Flash Attention

# experiments/my_exp/gpt_lab/nn_modules/sequence_mixing/causal_self_attention.py
class CausalSelfAttention(FlashAttentionBase):
    """Drop-in replacement using Flash Attention."""
    pass

# Same import, now uses your version:
from gpt_lab.nn_modules.sequence_mixing import CausalSelfAttention
# Gets your implementation automatically
```

### Scenario: Adding a New Feature

```python
# experiments/my_exp/gpt_lab/train_loops/gradient_smoothing.py
def run_training(model, optimizer, train_loader, smoothing_factor=0.1, **kwargs):
    """Custom training loop feature."""
    # Implementation
    pass

# Use it with smart_train:
smart_train(
    model, optimizer, train_loader,
    smoothing_factor=0.5  # Automatically discovered and used
)
```

## Inspecting the System

Debug the active configuration:

```bash
# Show active roots and package paths
python CLIs/print_active_paths.py -v

# Output:
# current_experiment: my_exp
# active_experiments: ['my_exp']
# active_packs: ['nlp']
# ordered_roots:
#   - experiments/my_exp/gpt_lab
#   - catalogs/packs/nlp/gpt_lab
#   - catalogs/core/gpt_lab
```

## Advanced Topics

### Custom Catalog Types

You can add new catalog types beyond the standard ones:

```python
# experiments/my_exp/gpt_lab/my_custom_catalog/
```

The bootstrap process will automatically create `gpt_lab.my_custom_catalog` namespace.

### Root Markers

Each root should have a `.gpt_lab_root` marker file at the repository root for validation:

```bash
python CLIs/validate_layout.py
```

## Summary

GPT-Lab's architecture enables:
- **Modular development**: Build components independently
- **Easy experimentation**: Override anything without breaking shared code
- **Reproducible research**: Explicit activation and precedence
- **Collaborative work**: Share packs without forcing adoption
- **Scalable organization**: Add experiments and packs without restructuring

The namespace bootstrapping approach makes this possible by leveraging Python's module system rather than fighting against it.
