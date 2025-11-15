# GPT-Lab

A framework for modular, testable, and reproducible ML research. GPT-Lab helps researchers build experiments with strong reproducibility guarantees while enabling rapid iteration.

# WARNING

This repo is in early alpha and frequently undergoing major restructuring.
It is my belief that we are somewhat close to the final structure but the implementation is definitely not as clean as it could be.
At the vary least, you should be able to rely on working on experiments inside `experiments/<experiment_name>/` to stay consistent in structure, but the same cannot be said for experiments in separate repos or git submodules.

## Key Features

- **Modular Catalogs**: Composable components for models, optimizers, train loops, and data sources
- **Namespace Bootstrapping**: Flexible catalog activation across experiments, packs, and core
- **Reproducibility**: Git tracking, RNG state management, and experiment restoration
- **Testing**: Automated discovery-based tests for all catalog items
- **Interactive Tools**: Marimo notebooks for analysis and benchmarking

## Quick Start

### Installation

```bash
# Install with all development dependencies
pip install -e '.[dev]'

# Or install specific extras
pip install -e '.[nlp]'  # NLP pack
pip install -e '.[cv]'   # CV pack (planned)
```

### Run Tests

```bash
pytest
```

See [docs/testing.md](docs/testing.md) for details.

### Create an Experiment

```bash
python CLIs/scaffold_experiment.py my_experiment
cd experiments/my_experiment
python main.py
```

## Documentation

**Full documentation is available in the `docs/` directory.**

View locally with MkDocs:

```bash
pip install mkdocs
mkdocs serve
```

Then open `http://127.0.0.1:8000`

### Documentation Structure

- **Concepts**: [Architecture](docs/architecture.md) | [Configuration](docs/config.md) | [Testing](docs/testing.md)
- **Components**: [Logger](docs/components-logger.md) | [Checkpointer](docs/components-checkpointer.md) | [Reproducibility](docs/components-reproducibility.md) | [Device](docs/components-device.md) | [Distributed](docs/components-distributed.md) | [Configuration](docs/components-configuration.md)
- **CLIs**: [Scaffold](docs/cli-scaffold-experiment.md) | [Test All](docs/cli-pytest-all-experiments.md) | [Multi-Run](docs/cli-run-multi.md) | [Print Paths](docs/cli-print-active-paths.md) | [Validate](docs/cli-validate-layout.md)
- **Notebooks**: [Marimo Intro](docs/notebooks-about-marimo.md) | [Experiment Comparison](docs/notebooks-experiment-comparison.md) | [Benchmarks](docs/notebooks-nn_module-bench.md)
- **Catalogs**: [Train Loops](docs/catalogs-train-loops.md) | [Modules](docs/catalogs-modules.md) | [Optimizers](docs/catalogs-optimizers.md) | [Models](docs/catalogs-models.md)
- **Packs**: [Core](docs/packs-core.md) | [NLP](docs/packs-nlp.md) | [CV](docs/packs-cv.md)

## Example Usage

### Basic Experiment

```python
import argparse
from gpt_lab.configuration import get_config
from gpt_lab.reproducibility import ReproducibilityManager
from gpt_lab.distributed import DistributedManager
from gpt_lab.logger import setup_experiment_logging
from gpt_lab.train_loops import smart_train

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
            
            # Your training code
            smart_train(
                model=model,
                train_loader=train_loader,
                optimizer=optimizer,
                num_epochs=config['num_epochs']
            )

if __name__ == "__main__":
    main()
```

### Activating Catalogs

Via environment variables:

```bash
export GPT_LAB_CURRENT_EXPERIMENT=nano_gpt
export GPT_LAB_ACTIVE_PACKS=nlp
```

Via YAML files:

```yaml
# experiments/my_exp/gpt_lab.yaml
include_experiments: []
include_packs: ['nlp']
```

Debug activation:

```bash
python CLIs/print_active_paths.py -v
```

## Architecture

GPT-Lab organizes code into catalogs under a unified `gpt_lab.*` namespace with configurable precedence:

1. **Current experiment** (highest precedence)
2. **Active experiments**
3. **Active packs**
4. **Core** (lowest precedence, always active)

Each level can override or extend components from lower levels.

See [docs/architecture.md](docs/architecture.md) for details.

## Repository Structure

```text
├── src/gpt_lab/          # Main package source
├── tests/                # Test discovery and execution
├── experiments/          # Experiment catalog
├── catalogs/
│   ├── core/            # Core components (always active)
│   └── packs/           # Domain-specific packs (nlp, cv)
├── CLIs/                # Command-line tools
├── notebooks/           # Marimo notebooks for analysis
├── docs/                # Documentation
└── pyproject.toml       # Package configuration
```

## Development

### Running Tests

```bash
# All tests
pytest

# Specific experiment
python CLIs/pytest_all_experiments.py --include nano_gpt

# With coverage
pytest --cov=src/gpt_lab --cov-report=html
```

See [docs/testing.md](docs/testing.md) for details.

### Benchmarking

```bash
# Run benchmarks
python -m gpt_lab.nn_modules.catalog_benchmark
python -m gpt_lab.optimizers.catalog_benchmark

# View results
marimo edit notebooks/nn_modules_bench.py
marimo edit notebooks/optimizers_bench.py
```

### Contributing

1. Create feature branch
2. Add tests for new components
3. Update documentation
4. Run full test suite
5. Submit pull request

See individual documentation files for contributing guidelines for specific components.

## License

See [LICENSE](LICENSE) for details.

## Links

- **Documentation**: Run `mkdocs serve` and visit `http://127.0.0.1:8000`
- **Issues**: Report bugs and request features via GitHub issues
- **Examples**: See `experiments/` directory for working examples