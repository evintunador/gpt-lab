# Benchmarks

- `gpt_lab.benchmarks.runner.BenchmarkRunner`: abstract base for benchmarks.
  - Use `@register_handler('type')` on model methods to declare which handler processes a given benchmark type.
  - Implement `_initialize_metrics`, `_process_results_batch`, `_compute_final_metrics` in subclass.
  - `run(dataset, batch_size=1, limit=None) -> metrics`: iterates with progress bar and aggregates metrics.
