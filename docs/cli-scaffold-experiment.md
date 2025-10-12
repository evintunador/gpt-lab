# Scaffold Experiment CLI

The `scaffold_experiment.py` script creates a new experiment with the proper directory structure and template files.

## Overview

This CLI tool:
- Creates experiment directory structure
- Generates template `gpt_lab.yaml` configuration
- Creates placeholder `main.py`
- Sets up `gpt_lab/` and `artifacts/` directories
- Ensures proper structure for catalog discovery

## Usage

```bash
python CLIs/scaffold_experiment.py my_new_experiment
```

## Generated Structure

### Directory Layout

```
experiments/my_new_experiment/
├── gpt_lab.yaml              # Catalog configuration
├── gpt_lab/                  # Catalog root for this experiment
│   └── artifacts/            # Compiled train loops, benchmarks
│       └── .gitkeep
└── main.py                   # Entry point script
```

After scaffolding, add catalog items:

```
experiments/my_transformer_exp/
├── gpt_lab.yaml
├── gpt_lab/
│   ├── artifacts/
│   ├── nn_modules/
│   │   └── my_custom_attention.py
│   ├── models/
│   │   └── my_model.py
│   └── train_loops/
│       └── my_training_loop.py
└── main.py
```

### `gpt_lab.yaml` Template

```yaml
include_experiments: []
include_packs: []
```

This file configures which experiments and packs to include when this experiment is active.

Edit `gpt_lab.yaml` to include other experiments or packs:

```yaml
# experiments/my_transformer_exp/gpt_lab.yaml
include_experiments: []  # Other experiments to include
include_packs: ['nlp']   # Include NLP pack for data sources
```

### `main.py` Template

```python
if __name__ == '__main__':
    print('Hello from experiment: my_new_experiment')
```

Replace with your experiment code.

## Testing

To test this specific experiment, either setup env variables:

```bash
GPT_LAB_CURRENT_EXPERIMENT=my_exp pytest src/gpt_lab/tests/ -v
```

OR use the repo-wide tool:

```bash
# Test your experiment
python CLIs/pytest_all_experiments.py --include my_new_experiment
```