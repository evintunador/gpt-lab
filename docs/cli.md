# CLI Tools

This repo includes helper scripts under `tools/CLIs/`.

## Print active paths

Show active context and resolved package paths:
```bash
python tools/CLIs/print_active_paths.py -v
```

## Run pytest once per experiment

Runs tests with each experiment isolated (used by nightly CI):
```bash
python tools/CLIs/pytest_all_experiments.py --pytest-args -q
```

## Run multiple commands (sweeps)

`run_multi.py` can expand parameter grids and execute runs sequentially or in parallel:
```bash
python tools/CLIs/run_multi.py --config my_sweep.yaml
```
YAML config defines `name`, `command` (type: `python`/`torchrun`/`sbatch`), `parameters` (grid), and `execution`.

## Scaffold a new experiment

Create `experiments/<name>/` with boilerplate:
```bash
python tools/CLIs/scaffold_experiment.py my_new_exp
```

## Validate repo layout

Validate the repository’s expected structure and YAML formatting:
```bash
python tools/CLIs/validate_layout.py
```
