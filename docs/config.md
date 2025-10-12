# Configuration

GPT-Lab uses two mechanisms to define which roots are active and their precedence: **environment variables** and **YAML files**.

## Quick Reference

| Aspect | Environment Variables | YAML Files |
|--------|----------------------|-------------|
| Precedence | Highest | Fallback |
| Scope | Runtime/session | Project/experiment default |
| Files | N/A | `gpt_lab.yaml` (repo or experiment) |
| Best for | Testing, CI/CD, overrides | Default configurations |

## Environment Variables

Environment variables provide the highest precedence for controlling activation.

### Available Variables

**`GPT_LAB_CURRENT_EXPERIMENT`**
- The primary experiment to activate
- Default: Inferred from current working directory
- Example: `export GPT_LAB_CURRENT_EXPERIMENT=nano_gpt`

**`GPT_LAB_ACTIVE_EXPERIMENTS`**
- Comma-separated list of additional experiments
- Appended after current experiment in precedence order
- Example: `export GPT_LAB_ACTIVE_EXPERIMENTS=nano_gpt,modded_nano_gpt`

**`GPT_LAB_ACTIVE_PACKS`**
- Comma-separated list of packs to include
- Added after experiments, before core
- Example: `export GPT_LAB_ACTIVE_PACKS=nlp,cv`

### Usage Examples

**Single experiment:**
```bash
export GPT_LAB_CURRENT_EXPERIMENT=nano_gpt
python experiments/nano_gpt/main.py
```

**Experiment with pack:**
```bash
export GPT_LAB_CURRENT_EXPERIMENT=my_nlp_exp
export GPT_LAB_ACTIVE_PACKS=nlp
python experiments/my_nlp_exp/main.py
```

**Multiple experiments:**
```bash
# Use my_exp as primary, but also include nano_gpt's components
export GPT_LAB_CURRENT_EXPERIMENT=my_exp
export GPT_LAB_ACTIVE_EXPERIMENTS=my_exp,nano_gpt
export GPT_LAB_ACTIVE_PACKS=nlp
```

**Disable all activation (core only):**
```bash
export GPT_LAB_CURRENT_EXPERIMENT=none
export GPT_LAB_ACTIVE_EXPERIMENTS=
export GPT_LAB_ACTIVE_PACKS=
pytest  # Tests only core components
```

### Environment Variable Precedence

When multiple variables are set:

```bash
export GPT_LAB_CURRENT_EXPERIMENT=my_exp
export GPT_LAB_ACTIVE_EXPERIMENTS=my_exp,other_exp
export GPT_LAB_ACTIVE_PACKS=nlp

# Resulting precedence:
# 1. my_exp (current)
# 2. my_exp (from active list - will be deduplicated)
# 3. other_exp
# 4. nlp pack
# 5. core
```

## YAML Files

YAML files provide default configurations that environment variables can override.

### File Locations

**Repo-level** (`gpt_lab.yaml` at repository root):
- Defines global defaults for all experiments
- Rarely used in practice

**Experiment-level** (`experiments/<name>/gpt_lab.yaml`):
- Defines defaults for that specific experiment
- Most common use case

### YAML Structure

```yaml
include_experiments: []  # List of experiments to include
include_packs: []        # List of packs to include
```

### Examples

**Experiment with NLP pack:**
```yaml
# experiments/my_nlp_exp/gpt_lab.yaml
include_experiments: []
include_packs: ['nlp']
```

**Experiment building on another:**
```yaml
# experiments/improved_nano_gpt/gpt_lab.yaml
include_experiments: ['nano_gpt']  # Include nano_gpt's components
include_packs: ['nlp']
```

**Complex composition:**
```yaml
# experiments/multi_modal/gpt_lab.yaml
include_experiments: ['nano_gpt', 'vision_transformer']
include_packs: ['nlp', 'cv']
```

### Resolution Order

For each list (experiments and packs), the resolution order is:

1. **Environment variable** (highest)
2. **Experiment YAML**
3. **Repo YAML** (lowest)

### Example Resolution

```yaml
# gpt_lab.yaml (repo root)
include_packs: ['nlp']

# experiments/my_exp/gpt_lab.yaml
include_packs: ['nlp', 'cv']
```

```bash
# With no ENV set:
# Uses experiment YAML: ['nlp', 'cv']

# With ENV set:
export GPT_LAB_ACTIVE_PACKS=cv
# ENV overrides YAML: ['cv']
```

## Directory Inference

If `GPT_LAB_CURRENT_EXPERIMENT` is not set, GPT-Lab infers it from the current working directory:

```bash
cd experiments/nano_gpt
python main.py
# Automatically sets current_experiment=nano_gpt
```

This makes running experiments intuitive without explicit configuration.

## Precedence Summary

Complete precedence order for activated roots:

```python
# Example configuration:
GPT_LAB_CURRENT_EXPERIMENT=my_exp
GPT_LAB_ACTIVE_EXPERIMENTS=my_exp,other_exp
GPT_LAB_ACTIVE_PACKS=nlp,cv

# Resulting precedence (highest to lowest):
ordered_roots = [
    'experiments/my_exp/',        # Current experiment
    'experiments/other_exp/',     # Additional experiment
    'catalogs/packs/nlp/',        # First pack
    'catalogs/packs/cv/',         # Second pack
    'catalogs/core/'              # Core (always last)
]
```

## Inspecting Active Configuration

### Print Active Paths CLI

```bash
python CLIs/print_active_paths.py -v
```

**Output:**
```
repo_root: /Users/you/repos/gpt-lab
current_experiment: my_exp
active_experiments: ['my_exp', 'other_exp']
active_packs: ['nlp', 'cv']

ordered_roots:
  - /Users/you/repos/gpt-lab/experiments/my_exp/gpt_lab
  - /Users/you/repos/gpt-lab/experiments/other_exp/gpt_lab
  - /Users/you/repos/gpt-lab/catalogs/packs/nlp/gpt_lab
  - /Users/you/repos/gpt-lab/catalogs/packs/cv/gpt_lab
  - /Users/you/repos/gpt-lab/catalogs/core/gpt_lab

gpt_lab.nn_modules.__path__:
  - /Users/you/repos/gpt-lab/experiments/my_exp/gpt_lab/nn_modules
  - /Users/you/repos/gpt-lab/experiments/other_exp/gpt_lab/nn_modules
  - /Users/you/repos/gpt-lab/catalogs/packs/nlp/gpt_lab/nn_modules
  - /Users/you/repos/gpt-lab/catalogs/core/gpt_lab/nn_modules
```

### From Python

```python
from gpt_lab.catalog_bootstrap import get_active_context

ctx = get_active_context()
print("Current experiment:", ctx['current_experiment'])
print("Active packs:", ctx['active_packs'])
print("Ordered roots:", ctx['ordered_roots'])
```

## Use Cases

### Development

```bash
# Work on your experiment with full activation
cd experiments/my_exp
python main.py
```

### Testing

```bash
# Test specific experiment in isolation
export GPT_LAB_CURRENT_EXPERIMENT=nano_gpt
export GPT_LAB_ACTIVE_EXPERIMENTS=nano_gpt
pytest

# Test with pack
export GPT_LAB_ACTIVE_PACKS=nlp
pytest
```

### CI/CD

```yaml
# .github/workflows/test.yml
- name: Test core only
  run: |
    export GPT_LAB_CURRENT_EXPERIMENT=none
    export GPT_LAB_ACTIVE_EXPERIMENTS=
    export GPT_LAB_ACTIVE_PACKS=
    pytest

- name: Test each experiment
  run: python CLIs/pytest_all_experiments.py
```

### Debugging

```bash
# Check what's active
python CLIs/print_active_paths.py -v

# Test with different configurations
export GPT_LAB_ACTIVE_PACKS=nlp
python CLIs/print_active_paths.py

export GPT_LAB_ACTIVE_PACKS=cv
python CLIs/print_active_paths.py
```

## Best Practices

### 1. Use YAML for Defaults

Set experiment defaults in `gpt_lab.yaml`:

```yaml
# experiments/my_exp/gpt_lab.yaml
include_packs: ['nlp']  # Default pack for this experiment
```

### 2. Use ENV for Overrides

Override for testing or special runs:

```bash
# Temporarily disable pack
export GPT_LAB_ACTIVE_PACKS=
python main.py
```

### 3. Document Requirements

```yaml
# experiments/my_exp/gpt_lab.yaml
# This experiment requires the NLP pack for tokenizers and benchmarks
include_packs: ['nlp']
```

### 4. Avoid Circular Dependencies

Don't have experiments include each other circularly:

```yaml
# Bad: experiments/a/gpt_lab.yaml
include_experiments: ['b']

# experiments/b/gpt_lab.yaml
include_experiments: ['a']  # Circular!
```

### 5. Verify Before Running

```bash
python CLIs/print_active_paths.py -v
# Check output looks correct
python main.py
```

## Troubleshooting

### Unexpected Components

**Problem**: Wrong version of a component is being used.

**Solution**: Check precedence:
```bash
python CLIs/print_active_paths.py -v
# Look for which root has the component
```

### Components Not Found

**Problem**: `ImportError: cannot import name 'MyModule'`

**Solution**: Verify activation includes the root with that component:
```bash
python CLIs/print_active_paths.py -v
# Check if the relevant pack/experiment is active
```

### Tests Finding Wrong Items

**Problem**: Tests discover components from unexpected roots.

**Solution**: Explicitly set activation for tests:
```bash
export GPT_LAB_CURRENT_EXPERIMENT=my_exp
export GPT_LAB_ACTIVE_EXPERIMENTS=my_exp
export GPT_LAB_ACTIVE_PACKS=nlp
pytest
```

## Related Documentation

- [Architecture](architecture.md) - How roots and precedence work
- [Testing](testing.md) - How activation affects test discovery
- [Print Active Paths CLI](cli-print-active-paths.md) - Debug activation
