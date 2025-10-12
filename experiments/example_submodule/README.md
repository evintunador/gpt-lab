# Example Submodule Experiment

This shows how an experiment can live as a separate git repo and be included here as a submodule.

## As a submodule inside this monorepo

1) External repo layout:
```
experiments/example_submodule/
  gpt_lab/
    artifacts/
  gpt_lab.yaml
  main.py
```
2) Add as submodule:
```bash
git submodule add <git-url> experiments/example_submodule
```
3) Activate and run:
```bash
export GPT_LAB_CURRENT_EXPERIMENT=example_submodule
export GPT_LAB_ACTIVE_EXPERIMENTS=example_submodule
python tools/CLIs/print_active_paths.py -v
```

## Standalone usage

If checking out just the experiment repo, point GPT_LAB_ROOT to this monorepo:
```bash
export GPT_LAB_ROOT=/abs/path/to/gpt-lab
export GPT_LAB_CURRENT_EXPERIMENT=example_submodule
export GPT_LAB_ACTIVE_EXPERIMENTS=example_submodule
python experiments/example_submodule/main.py
```

Artifacts default to `experiments/example_submodule/gpt_lab/artifacts/`.
