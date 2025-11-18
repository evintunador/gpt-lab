# Reproducibility

The `gpt_lab.reproducibility` module captures experiment state (git, environment, RNG, invocation) and helps persist artifacts reliably with optional storage backends and graceful shutdown.

## Overview

The reproducibility component is designed to:
- Capture complete git state for the main repo, its git submodules, and any enclosing superprojects (commit hash, branch, remote URL, dirty status, GitHub URL when possible)
- Save patches of uncommitted and untracked changes for dirty repositories (main repo, submodules, superprojects)
- Snapshot software, hardware, distributed, and invocation state into structured JSON files
- Capture initial and final RNG states for `torch`, `numpy`, and Python's `random`
- Create the output directory you specify (main process only)
- Optionally store artifacts with customizable backup storage backends (catalog: `gpt_lab.backup_storage_backends`)
- Optionally notify daemon hooks on run start/end (catalog: `gpt_lab.daemon_hooks`)
- Gracefully handle interrupts (SIGINT/SIGTERM) and still upload artifacts when configured

## Key Components

### `ReproducibilityManager`

A context manager that captures experiment state and manages artifact storage:

```python
from gpt_lab.reproducibility import ReproducibilityManager

with ReproducibilityManager(
    output_dir="./experiments/my_exp/runs",
    backup_storage_backend=None,  # optional catalog item
    daemon_hook=None,             # optional catalog item
) as repro:
    # repro.output_dir: absolute path you provided
    # repro.git_info: dict with git metadata (including submodules/superprojects)
    # repro.software_environment: software/package info
    # repro.runtime_environment: hardware / OS / distributed info
    # repro.run_invocation: argv + important env vars
    # Your experiment code here
    train_model()
```

**Parameters:**
- `output_dir`: Root directory for experiment outputs (created only by the main process)
- `backup_storage_backend`: Optional backup storage backend for artifacts (uploads on exit when provided). Items come from the `gpt_lab.backup_storage_backends` catalog.
- `daemon_hook`: Optional hook for liveness monitoring. Items come from the `gpt_lab.daemon_hooks` catalog.

**Behavior:**

On **entry** (`__enter__`):
1. Creates the provided output directory and a nested `reproducibility/` directory (main process only; determined via `gpt_lab.distributed.is_main_process()`).
2. Captures git state (commit, branch, remote URL, GitHub URL when derivable) for the main repo.
3. Detects whether the working directory is dirty and, if so, saves a patch of uncommitted changes.
4. Captures git metadata for all git submodules (recursively) and any enclosing superprojects, saving patches for dirty ones as well.
5. Saves `reproducibility/git_info.json` with git metadata.
6. Snapshots the software environment to `reproducibility/software_environment.json`.
7. Snapshots the runtime environment (OS, devices, distributed topology, CUDA/driver info) to `reproducibility/runtime_environment.json`.
8. Captures initial RNG state to `reproducibility/rng_state_initial.pt`.
9. Captures invocation details (argv + filtered env) to `reproducibility/run_invocation.json`.
10. Calls `on_run_start` on the provided `daemon_hook` if provided.
11. Registers signal handlers for `SIGINT` (Ctrl+C) and `SIGTERM` to enable graceful shutdown.

On **exit** (`__exit__`):
1. Captures final RNG state to `reproducibility/rng_state_final.pt`.
2. Calls `on_run_end` on the provided `daemon_hook` if provided.
3. Uploads artifacts to the backup storage backend if provided.
4. Works even if the experiment exits with an error or is interrupted.
5. Restores original signal handlers to avoid side effects.
6. Synchronizes processes using `gpt_lab.distributed.barrier()` so non-main processes wait for the main process to finish uploads (no-op in single-process mode).

**Created Directory Structure:**
```
experiments/my_exp/runs/<specific_run_id>/
├── reproducibility/
│   ├── git_info.json
│   ├── software_environment.json
│   ├── runtime_environment.json
│   ├── run_invocation.json
│   ├── rng_state_initial.pt
│   ├── rng_state_final.pt
│   ├── uncommitted_changes.patch                         (if main repo is dirty)
│   ├── uncommitted_changes.submodule.<path>.patch        (for dirty submodules)
│   └── uncommitted_changes.superproject.<idx>.<name>.patch (for dirty superprojects)
├── checkpoints/
├── logs/
└── ... (your experiment outputs)
```

### `git_info` Dictionary

The captured git information for the main repository includes:

```python
{
    "commit_hash": "a1b2c3d4e5f6...",
    "branch": "main",
    "remote_url": "git@github.com:user/repo.git",
    "github_url": "https://github.com/user/repo/commit/a1b2c3d...",
    "git_is_dirty": False,
    "patch_file": "/path/to/.../reproducibility/uncommitted_changes.patch",  # present only if dirty
    "submodules": [
        {
            "path": "external/lib",
            "repo_path": "/absolute/path/to/external/lib",
            "commit_hash": "...",
            "branch": "...",
            "remote_url": "...",
            "github_url": "...",
            "git_is_dirty": False,
            # Optional if dirty:
            # "patch_file": "/.../reproducibility/uncommitted_changes.submodule.external__lib.patch"
        },
        ...
    ],
    "superprojects": [
        {
            "path": "/absolute/path/to/superproject/root",
            "commit_hash": "...",
            "branch": "...",
            "remote_url": "...",
            "github_url": "...",
            "git_is_dirty": False,
            # Optional if dirty:
            # "patch_file": "/.../reproducibility/uncommitted_changes.superproject.0.superproject.patch",
            # And any submodules inside the superproject:
            # "submodules": [...]
        },
        ...
    ],
}
```

### Environment and Invocation Snapshots

Instead of ad-hoc calls, `ReproducibilityManager` persists structured snapshots under `reproducibility/` and exposes them via properties:

```python
with ReproducibilityManager(output_dir="./runs/my_exp") as repro:
    # Software environment (Python, package versions, framework settings)
    sw = repro.software_environment
    # {
    #   "python_version": "...",
    #   "package_versions": { "torch": "x.y.z", "numpy": "...", ... },
    #   "torch_repro_settings": { ... determinism / precision knobs ... },
    # }

    # Runtime environment (OS, devices, distributed topology, CUDA runtime)
    rt = repro.runtime_environment
    # {
    #   "cuda_available": true/false,
    #   "device_count": int,
    #   "devices": ["GPU names..."],
    #   "device_properties": [...],
    #   "distributed": {...},
    #   "os": {...},
    #   "cuda_runtime": {...},
    # }

    # Run invocation (argv + filtered env vars)
    inv = repro.run_invocation
    # {
    #   "argv": [...],
    #   "env": {
    #       "CUDA_VISIBLE_DEVICES": "...",
    #       "WORLD_SIZE": "...",
    #       "PYTHONHASHSEED": "...",
    #       "PYTHONPATH": "...",
    #       "VIRTUAL_ENV": "...",
    #       ...
    #   },
    # }
```

### RNG State Helpers

`ReproducibilityManager` records RNG state at entry and exit and provides helpers for manual snapshots:

```python
with ReproducibilityManager(output_dir="./runs/my_exp") as repro:
    # Initial RNG state captured at context entry
    initial_rng = repro.initial_rng_state

    # Current RNG state (on demand)
    current_rng = repro.get_rng_states()

    # Restore RNG state
    repro.set_rng_states(initial_rng)

    # Final RNG state will be saved on exit to:
    # ./runs/my_exp/<run_id>/reproducibility/rng_state_final.pt
    final_rng = repro.final_rng_state  # None until after __exit__ runs
```

### Backup Storage Backends (Catalog)

Backups are handled by catalog items implementing a standard interface. The manager calls:

```python
backup_storage_backend.upload(source_dir="./original_run")
```

And for restoration:

```python
backup_storage_backend.download(destination_dir="./restored_run")
```

For more info on Backup Storage Backends, see [this doc](catalogs-backup-storage-backends.md).

## Helper Functions

## Usage Examples

### Basic Experiment Setup

```python
import logging
from gpt_lab.reproducibility import ReproducibilityManager
# from gpt_lab.logger import setup_experiment_logging  # optional

with ReproducibilityManager(output_dir="./runs") as repro:
    # Setup logging
    # log_dir = os.path.join(repro.output_dir, "logs")
    # setup_experiment_logging(log_dir, rank=0, is_main_process=True)
    
    logger = logging.getLogger(__name__)
    
    # Log experiment directory
    logger.info(f"Output directory: {repro.output_dir}")
    
    # Run experiment
    train_model()
```

## Best Practices

### 1. Always Use ReproducibilityManager

Wrap your experiment entry point:

```python
def main():
    with ReproducibilityManager(output_dir="./runs") as repro:
        # All experiment code here
        train_model()

if __name__ == "__main__":
    main()
```

### 2. Commit Before Running Experiments

```bash
git add .
git commit -m "Experiment: testing new architecture"
python main.py
```

This ensures clean git state and better experiment tracking.

### 3. Include Git Info in All Checkpoints

```python
save_checkpoint(
    ...,
    metadata={
        "epoch": epoch,
        "git_info": repro.git_info  # Always include!
    }
)
```

## Testing

Comprehensive tests are in `tests/src/gpt_lab/test_reproducibility.py`:

```bash
pytest tests/src/gpt_lab/test_reproducibility.py -v
```

**Test coverage includes:**
- Clean repository state capture
- Dirty repository state capture (modified and untracked files)
- Patch file creation and application
- Storage backend upload/download
- Distributed training awareness
- Signal handling for graceful shutdown and artifact upload