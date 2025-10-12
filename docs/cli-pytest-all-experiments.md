# Pytest All Experiments CLI

The `pytest_all_experiments.py` script runs pytest for each experiment in isolation, ensuring that each experiment's catalog items are tested independently.

## Overview

This CLI tool:
- Discovers all experiments in the `experiments/` directory
- Runs pytest separately for each experiment with its own catalog items
- Aggregates results and reports failures
- Supports filtering experiments (include/exclude)
- Passes additional pytest arguments through to each run

## Usage

```bash
# Run tests for all experiments
python CLIs/pytest_all_experiments.py

# Run with pytest options
python CLIs/pytest_all_experiments.py --pytest-args -v -k test_specific

# Include only specific experiments
python CLIs/pytest_all_experiments.py --include nano_gpt modded_nano_gpt

# Exclude specific experiments
python CLIs/pytest_all_experiments.py --exclude custom_bpe
```

## How It Works

For each experiment:

1. Sets environment variables:
   ```bash
   GPT_LAB_CURRENT_EXPERIMENT=<experiment_name>
   GPT_LAB_ACTIVE_EXPERIMENTS=<experiment_name>
   ```

2. Runs pytest with any additional arguments:
   ```bash
   python -m pytest <extra_args>
   ```

3. Collects exit codes

4. Reports summary at the end

## Troubleshooting

### Some Experiments Fail

Check the output for specific failure messages:

```bash
python CLIs/pytest_all_experiments.py --pytest-args -v --tb=long
```

### Tests Don't Discover Items

Ensure each experiment has proper catalog structure and `gpt_lab.yaml`:

```bash
experiments/my_exp/
├── gpt_lab.yaml
├── gpt_lab/
│   ├── nn_modules/
│   ├── optimizers/
│   └── train_loops/
└── main.py
```

### Environment Variables Not Working

Verify environment is clean before running:

```bash
unset GPT_LAB_CURRENT_EXPERIMENT
unset GPT_LAB_ACTIVE_EXPERIMENTS
unset GPT_LAB_ACTIVE_PACKS

python CLIs/pytest_all_experiments.py
```

