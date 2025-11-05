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

# todo
## important / urgent

- [ ] design & build a mu-parametrization utility
- [ ] do first DAGSeq2DAGSeq experiment
    - [ ] reassess what we need after having actually used this system in DAGSeq2DAGSeq

### important / not-urgent

- [ ] fact check various inaccuracies in the documentation
- [ ] design and build a system for comparing performance between two experiments or/
and i guess different config settings within an experiment both directly and as a function of the performance per runtime/memory difference
    - [x] minimal proof of concept
    - [ ] time series
    - [ ] more adaptable to whatever's available in the experiments
- [ ] design & build hyperparameter search utility with an interface such that we can change out search algorithms later
- [ ] setup a docker container to develop in to ensure consistent behavior across systems
- [ ] add slurm capabilities to DistributedManager
- [ ] implement more advanced parallel abilities for `src/gpt_lab/nn_modules/` testing and benchmarking and general utils to help with the various types of parallelization, maybe in DistributedManager? maybe in its own ParallelizationManager?
- [ ] abstract out evaluation utilities. rn we've got `src/benchmarks/` which seems able to run benchmark datasets but i'd also like general evaluation metrics like perplexity to get recorded. maybe a benchmark is a specific type of evaluation that takes in an external dataset? does regular validation count as a type of evaluation? idk how this works
- [ ] build a tool to allow the experiment to dynamically increase or decrease the number of nodes it's taking up by periodically checking for outside requests. it'd have to effectively re-adjust gradient accumulation settings in order to make the experiment numerically equivalent to when it had more/fewer nodes. i guess it wouldn't have to be aware of VRAM utilization since we'd keep the micro batch size the same and only change number of nodes and number of gradient accumulation steps? i don't think this would have to be aware of the gradient accumulation atomic feature; you'd just need to tell it which argument is the right one. is kinda ugly that people are roughly restricted to powers of 2 at that point. also this would have to overwrite whatever "waiting in line" system submitit has going on in order to restart a given experiment but smaller and let it skip forward in line. also ugly af thinking about resuming from the most recent checkpoint ew. not sure how feasible this is but i feel like it's necessary eventually

### not important / urgent

### not important / not urgent

- [ ] figure out a way to combine hyperparameter search, mu-parameterization, and model size & gpu vram awareness to allow for a model to scale itself up. this might be asking too much
- [ ] build a profiling system, likely for experiments themselves since what you care about at the end of the day is the full training loop's speed. hopefully i can use pytorch's built-in (it has one right?)
- [ ] find a cooler name for the repo
    - posture (bc it's helping you keep "good posture" when doing experiments)
    - {ml/dl/research/experiment/?}_harness
    - {ml/dl/?}-lab
    - just "lab"?
- [ ] design and build a wrapper around other general experiment utilities that need to be called at initialization to make the repo easier to use? do we even have enough stuff for that to be worth it?
- [ ] setup a "this atomic feature is a superset of atomic feature x" system that saves some context length for the LLM which should hopefully help both performance and costs
- [ ] abstract out some of what's in `src/gpt_lab/train_loops/` into `src/gpt_lab/
llm_code_compiler/` and find other use cases for our llm compiler system
- [ ] go around the repo looking for shared utilities across different catalog types that can be abstracted out
- [ ] revisit older project components that may have not been designed optimally (I'm particularly thinking of `src/gpt_lab/nn_modules/`)
- [ ] reduce duplicate dependencies (eg. we have both plotly & matplotlib)
- [x] move reproducibility.py's inbuilt CLI tool into CLIs/; same for all other CLIs inside src/
- [ ] make configuration.py not require any input argparser at all so users can rely entirely on config if they want
- [ ] make the example submodule experiment an actual submodule
- [ ] make a way to directly import the custom logger rather than having to use the setup function
- [ ] edit GLU to use our own custom version of LigerKernel that supports more activation functions
- [ ] move nn_module backup test discovery inside primary test discovery function & maybe get rid of it entirely
- [ ] improve attribute names and standardize all test & bench configs (hint: rename 'output_validator' to 'output_validator_fn' in nn.Modules bulk testing)
- [ ] write equivalents of `to_device` and `to_dtype` but for `.clone()` and `.detach()`
- [ ] separate core src/ importables from tools only used by catalog
- [ ] build system for updating a model's checkpoints as they change over time
- [ ] make checkpointer torch.compile & ddp aware
- [x] make logs also human readable
- [x] remove log requirement that you work inside experiments/
- [x] add support for external storage backup daemon to account for stalled experiments
- [ ] move reproducibility's storage backup backend ABC & daemon ABC over to the catalog system
- [ ] add non-grid indivdiual-command ability to the multi-run configs
- [x] move tests from src/gpt_lab/tests/ to tests/