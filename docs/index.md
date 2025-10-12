# GPT-Lab Documentation

Welcome to GPT-Lab, a framework for modular, testable, and reproducible ML research.

## What is GPT-Lab?

GPT-Lab helps researchers build experiments with:

- **Modular Catalogs**: Composable components (`nn_modules`, `optimizers`, `train_loops`, `benchmarks`, `data_sources`, `models`)
- **Namespace Bootstrapping**: Flexible layering of experiments, packs, and core components
- **Strong Reproducibility**: Git tracking, RNG state management, and experiment restoration
- **Automated Testing**: Discovery-based tests for all catalog items
- **Interactive Analysis**: Marimo notebooks for comparing results and benchmarking

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/evintunador/gpt-lab.git
cd gpt-lab

# Install for development
pip install -e '.[dev]'

# Install with specific packs
pip install -e '.[nlp]'  # NLP components
pip install -e '.[cv]'   # Computer Vision (planned)
```

### Run Tests

```bash
# Run all tests
pytest

# Run specific catalog tests
pytest src/gpt_lab/nn_modules/tests/ -v
pytest src/gpt_lab/optimizers/tests/ -v
```

### Create Your First Experiment

```bash
# Scaffold a new experiment
python CLIs/scaffold_experiment.py my_experiment

# Navigate to it
cd experiments/my_experiment

# Edit main.py with your training code
# Run it
python main.py
```

### Activate Catalogs

Experiments and packs are activated via environment variables or YAML files:

**Environment variables:**
```bash
export GPT_LAB_CURRENT_EXPERIMENT=nano_gpt
export GPT_LAB_ACTIVE_PACKS=nlp
```

**YAML file** (`experiments/my_experiment/gpt_lab.yaml`):
```yaml
include_experiments: []
include_packs: ['nlp']
```

**Debug activation:**
```bash
python CLIs/print_active_paths.py -v
```

## Example: Complete Training Script

```python
import argparse
from gpt_lab.configuration import get_config
from gpt_lab.reproducibility import ReproducibilityManager
from gpt_lab.distributed import DistributedManager
from gpt_lab.logger import setup_experiment_logging
from gpt_lab.train_loops import smart_train
import logging

def main():
    parser = argparse.ArgumentParser()
    config = get_config(parser)
    
    with DistributedManager() as dist:
        dist.set_seed(config['seed'])
        
        with ReproducibilityManager(
            output_dir=config['output_dir'],
            is_main_process=dist.is_main_process
        ) as repro:
            setup_experiment_logging(
                log_dir=f"{repro.output_dir}/logs",
                rank=dist.rank,
                is_main_process=dist.is_main_process
            )
            
            logger = logging.getLogger(__name__)
            logger.info("Starting experiment")
            
            # Smart train automatically composes training loop features
            smart_train(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=optimizer,
                num_epochs=config['num_epochs'],
                grad_accum_steps=4,
                mixed_precision=True,
                validate_every_n_epochs=1,
                checkpoint_every_n_epochs=5,
                save_best_model=True
            )

if __name__ == "__main__":
    main()
```

## Core Concepts

### Architecture

GPT-Lab uses **namespace bootstrapping** to compose catalogs from multiple sources with defined precedence:

1. **Current experiment** (highest precedence)
2. **Active experiments**
3. **Active packs**
4. **Core** (lowest precedence, always active)

See [Architecture](architecture.md) for details.

### Configuration

Two mechanisms control activation:
- **Environment variables** (highest precedence)
- **YAML files** (fallback defaults)

See [Configuration](config.md) for details.

### Testing

Tests automatically discover catalog items based on active roots. Run specific experiment tests or test all experiments in isolation.

See [Testing](testing.md) for details.

## Documentation Structure

### Concepts
- [Architecture](architecture.md) - How catalogs are organized and composed
- [Configuration](config.md) - Activation and precedence rules
- [Testing](testing.md) - Test discovery and execution

### Components
Importable tools for experiments:
- [Logger](components-logger.md) - Structured JSONL logging
- [Checkpointer](components-checkpointer.md) - Save/load training states
- [Reproducibility](components-reproducibility.md) - Git tracking and restoration
- [Device](components-device.md) - Hardware management
- [Distributed](components-distributed.md) - Multi-GPU training
- [Configuration](components-configuration.md) - YAML + CLI arg merging
- [Smart Training Loop](components-smart-train.md) - Composable training features

### CLIs
Command-line utilities:
- [Scaffold Experiment](cli-scaffold-experiment.md) - Create new experiments
- [Test All Experiments](cli-pytest-all-experiments.md) - Run tests in isolation
- [Run Multiple Experiments](cli-run-multi.md) - Hyperparameter sweeps
- [Print Active Paths](cli-print-active-paths.md) - Debug catalog activation
- [Validate Layout](cli-validate-layout.md) - Verify repository structure

### Notebooks
Interactive analysis tools:
- [About Marimo](notebooks-about-marimo.md) - Reactive notebooks introduction
- [Experiment Comparison](notebooks-experiment-comparison.md) - Compare runs
- [nn.Module Benchmarks](notebooks-nn_module-bench.md) - Module performance
- [Optimizer Benchmarks](notebooks-optimizers-bench.md) - Optimizer comparison

### Catalogs
Buildable component types:
- [Train Loops](catalogs-train-loops.md) - Atomic training features
- [nn.Modules](catalogs-modules.md) - Neural network components
- [Optimizers](catalogs-optimizers.md) - Training optimizers
- [Benchmarks](catalogs-benchmarks.md) - Evaluation frameworks
- [Models](catalogs-models.md) - High-level model wrappers

### Packs
Domain-specific component collections:
- [Core Pack](packs-core.md) - Foundation components (always active)
- [NLP Pack](packs-nlp.md) - Natural language processing
- [CV Pack](packs-cv.md) - Computer vision (planned)

## Next Steps

1. **Read [Architecture](architecture.md)** to understand how GPT-Lab organizes code
2. **Explore [Components](components-logger.md)** to learn about available tools
3. **Try the [CLI tools](cli-scaffold-experiment.md)** to create your first experiment
4. **Check [packs](packs-core.md)** to see what components are available

## Contributing

Contributions are welcome! Add components to:
- Core: `catalogs/core/gpt_lab/`
- Packs: `catalogs/packs/<pack>/gpt_lab/`
- Your experiment: `experiments/<your_exp>/gpt_lab/`

Each catalog type has specific conventions documented in its respective guide.

## License

See [LICENSE](../LICENSE) for details.
