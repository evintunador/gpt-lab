# Results Comparator (`gpt_lab.results_comparator`)

- Parse structured JSONL logs and extract metrics/hyperparameters per run.
- Strategy registry for aggregation: `last_value`, `best_value (min/max)`, `time_series`.
- Key functions:
  - `find_run_directories(globs, schema) -> list[str]`
  - `compare_runs(run_dirs, metric_defs, hparam_defs, schema) -> list[RunResult]`
  - `RunResult`: includes `metrics`, `hyperparameters`, and `git_info` if available.
