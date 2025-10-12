# Print Active Paths CLI

The `print_active_paths.py` script displays the current GPT-Lab catalog activation context, showing which experiments, packs, and catalog paths are active.

## Overview

This CLI tool:
- Shows active experiments and packs
- Lists all catalog roots in resolution order
- Displays resolved package `__path__`s for each catalog type
- Helps debug catalog activation issues
- Useful for understanding which catalog items will be discovered

## Usage

```bash
$ python CLIs/print_active_paths.py

repo_root: /Users/user/repos/gpt-lab
current_experiment: nano_gpt
active_experiments: ['nano_gpt']
active_packs: ['nlp']
ordered_roots:
 - /Users/user/repos/gpt-lab/experiments/nano_gpt/gpt_lab
 - /Users/user/repos/gpt-lab/catalogs/packs/nlp/gpt_lab
 - /Users/user/repos/gpt-lab/catalogs/core/gpt_lab
```

### Verbose Output

```bash
$ python CLIs/print_active_paths.py -v

repo_root: /Users/user/repos/gpt-lab
current_experiment: nano_gpt
active_experiments: ['nano_gpt']
active_packs: ['nlp']
ordered_roots:
 - /Users/user/repos/gpt-lab/experiments/nano_gpt/gpt_lab
 - /Users/user/repos/gpt-lab/catalogs/packs/nlp/gpt_lab
 - /Users/user/repos/gpt-lab/catalogs/core/gpt_lab

package __path__s:
gpt_lab.nn_modules:
  - /Users/user/repos/gpt-lab/experiments/nano_gpt/gpt_lab/nn_modules
  - /Users/user/repos/gpt-lab/catalogs/packs/nlp/gpt_lab/nn_modules
  - /Users/user/repos/gpt-lab/catalogs/core/gpt_lab/nn_modules
gpt_lab.optimizers:
  - /Users/user/repos/gpt-lab/catalogs/core/gpt_lab/optimizers
gpt_lab.train_loops:
  - /Users/user/repos/gpt-lab/experiments/nano_gpt/gpt_lab/train_loops
  - /Users/user/repos/gpt-lab/catalogs/core/gpt_lab/train_loops
gpt_lab.models:
  - /Users/user/repos/gpt-lab/experiments/nano_gpt/gpt_lab/models
  - /Users/user/repos/gpt-lab/catalogs/packs/nlp/gpt_lab/models
gpt_lab.data_sources:
  - /Users/user/repos/gpt-lab/catalogs/packs/nlp/gpt_lab/data_sources
gpt_lab.benchmarks:
  - /Users/user/repos/gpt-lab/catalogs/packs/nlp/gpt_lab/benchmarks
```

## Understanding the Output

### Repo Root

The repository root directory:
```
repo_root: /Users/user/repos/gpt-lab
```

### Current Experiment

The primary active experiment (from `GPT_LAB_CURRENT_EXPERIMENT` env var or CWD):
```
current_experiment: nano_gpt
```

### Active Experiments

List of all active experiments in order:
```
active_experiments: ['nano_gpt', 'modded_nano_gpt']
```

Higher in the list = higher precedence.

### Active Packs

List of active catalog packs:
```
active_packs: ['nlp', 'cv']
```

### Ordered Roots

All catalog roots in resolution order:
```
ordered_roots:
 - /path/to/experiments/nano_gpt/gpt_lab       (highest precedence)
 - /path/to/catalogs/packs/nlp/gpt_lab
 - /path/to/catalogs/core/gpt_lab              (lowest precedence)
```

Catalog items are discovered in this order. Items in higher roots shadow those in lower roots.

### Package __path__s (Verbose)

For each catalog type, shows all directories in the namespace package path:

```
gpt_lab.nn_modules:
  - /path/to/experiments/nano_gpt/gpt_lab/nn_modules
  - /path/to/catalogs/core/gpt_lab/nn_modules
```

Python will search these directories in order when importing from `gpt_lab.nn_modules`.

## Examples

### Verify Pack Activation

```bash
$ export GPT_LAB_ACTIVE_PACKS=nlp,cv
$ python CLIs/print_active_paths.py

active_packs: ['nlp', 'cv']
ordered_roots:
 - /path/to/catalogs/packs/nlp/gpt_lab
 - /path/to/catalogs/packs/cv/gpt_lab
 - /path/to/catalogs/core/gpt_lab
```

### Debug Catalog Discovery

```bash
# Why isn't my module being found?
$ python CLIs/print_active_paths.py -v

# Check if your experiment's gpt_lab directory is in ordered_roots
# Check if gpt_lab.nn_modules lists your experiment's directory
```

### Override with Environment Variables

```bash
$ export GPT_LAB_CURRENT_EXPERIMENT=modded_nano_gpt
$ export GPT_LAB_ACTIVE_PACKS=nlp
$ python CLIs/print_active_paths.py

current_experiment: modded_nano_gpt
active_packs: ['nlp']
# ...
```

### Check Multiple Experiments

```bash
$ export GPT_LAB_ACTIVE_EXPERIMENTS=nano_gpt,modded_nano_gpt
$ python CLIs/print_active_paths.py

active_experiments: ['nano_gpt', 'modded_nano_gpt']
ordered_roots:
 - /path/to/experiments/nano_gpt/gpt_lab
 - /path/to/experiments/modded_nano_gpt/gpt_lab
 - /path/to/catalogs/core/gpt_lab
```

### Core Only

```bash
# Unset all environment variables
$ unset GPT_LAB_CURRENT_EXPERIMENT
$ unset GPT_LAB_ACTIVE_EXPERIMENTS
$ unset GPT_LAB_ACTIVE_PACKS

# Run from outside any experiment directory
$ python CLIs/print_active_paths.py

current_experiment: None
active_experiments: []
active_packs: []
ordered_roots:
 - /path/to/catalogs/core/gpt_lab
```