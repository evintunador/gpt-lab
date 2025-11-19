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

A context manager that captures experiment state and manages artifact storage.

```python
from gpt_lab.reproducibility import ReproducibilityManager
from gpt_lab.distributed import is_main_process

# Example usage with explicit boolean
with ReproducibilityManager(
    output_dir="./experiments/my_exp/runs",
    is_main_process=is_main_process(),  # Required: bool or Callable[[], bool]
    backup_storage_backend=None,        # optional catalog item
    daemon_hook=None,                   # optional catalog item
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
- `output_dir`: Root directory for experiment outputs.
- `is_main_process`: **Required.** A `bool` or a `Callable[[], bool]` indicating if this is the main process.
    - `True`: Creates directories, uploads backups, runs daemon hooks.
    - `False`: Only saves local reproducibility artifacts (RNG state, env info) to its own subdirectory.
- `backup_storage_backend`: Optional backup storage backend for artifacts (uploads on exit when provided). Items come from the `gpt_lab.backup_storage_backends` catalog.
- `daemon_hook`: Optional hook for liveness monitoring. Items come from the `gpt_lab.daemon_hooks` catalog.

**Behavior:**

On **entry** (`__enter__`):
1.  **Guard Check:** If `is_main_process` is True, checks if `output_dir` already contains a `reproducibility/` folder. If so, raises `ValueError` to prevent overwriting previous experiment data.
2.  **Directory Creation:** Creates `output_dir` and a unique subdirectory for the current process: `reproducibility/node_<hostname>_pid_<pid>/`.
3.  **Artifact Capture (All Processes):**
    *   Captures git state (commit, branch, remote, dirty status, patches).
    *   Snapshots software environment (packages, versions).
    *   Snapshots runtime environment (OS, devices, distributed topology).
    *   Captures initial RNG state.
    *   Captures invocation details (argv + env).
    *   Saves all of the above into the process-unique subdirectory.
4.  **Main Process Actions:**
    *   Attempts to create a symlink `reproducibility/main` -> `reproducibility/node_<hostname>_pid_<pid>/` for convenience (best-effort).
    *   Calls `on_run_start` on the `daemon_hook`.
    *   Registers signal handlers for graceful shutdown.

On **exit** (`__exit__`):
1.  **Final RNG:** Captures final RNG state to `rng_state_final.pt` in the process-unique subdirectory.
2.  **Main Process Actions:**
    *   Calls `on_run_end` on the `daemon_hook`.
    *   Uploads artifacts to the `backup_storage_backend` if provided.
    *   Restores original signal handlers.
3.  **Synchronization:** Calls `barrier()` (using `torch.distributed` if initialized) to ensure non-main processes wait for the main process to finish uploads.

**Created Directory Structure:**
```
experiments/my_exp/runs/<specific_run_id>/
├── reproducibility/
│   ├── main -> node_hostA_pid_12345/                     (symlink, if supported)
│   ├── node_hostA_pid_12345/                             (Main Process)
│   │   ├── git_info.json
│   │   ├── software_environment.json
│   │   ├── runtime_environment.json
│   │   ├── run_invocation.json
│   │   ├── rng_state_initial.pt
│   │   ├── rng_state_final.pt
│   │   ├── uncommitted_changes.patch
│   │   └── ...
│   ├── node_hostA_pid_12346/                             (Worker Process 1)
│   │   └── ... (same structure, local state)
│   └── node_hostB_pid_67890/                             (Worker on another node)
│       └── ...
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
    "patch_file": "/path/to/.../reproducibility/.../uncommitted_changes.patch",  # present only if dirty
    "patch_file_hash": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", # SHA-256 of patch file
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
            # "patch_file": "/.../uncommitted_changes.submodule.external__lib.patch"
            # "patch_file_hash": "..."
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
            # "patch_file": "/.../uncommitted_changes.superproject.0.superproject.patch",
            # "patch_file_hash": "..."
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
with ReproducibilityManager(output_dir="./runs/my_exp", is_main_process=True) as repro:
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
with ReproducibilityManager(output_dir="./runs/my_exp", is_main_process=True) as repro:
    # Initial RNG state captured at context entry
    initial_rng = repro.initial_rng_state

    # Current RNG state (on demand)
    current_rng = repro.get_rng_states()

    # Restore RNG state
    repro.set_rng_states(initial_rng)

    # Final RNG state will be saved on exit to:
    # ./runs/my_exp/<run_id>/reproducibility/.../rng_state_final.pt
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
from gpt_lab.distributed import DistributedManager

# 1. Initialize Distributed Environment
with DistributedManager() as dist:
    
    # 2. Initialize Reproducibility Manager
    # Pass explicit is_main_process flag from your distributed manager or logic
    with ReproducibilityManager(
        output_dir="./runs/my_run_id",
        is_main_process=dist.is_main_process
    ) as repro:
        
        # Setup logging
        # log_dir = os.path.join(repro.output_dir, "logs")
        # setup_experiment_logging(log_dir, rank=dist.rank, is_main_process=dist.is_main_process)
        
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
    # ... setup distributed ...
    with ReproducibilityManager(output_dir="./runs", is_main_process=is_main) as repro:
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
- Distributed training awareness (simulated)
- Signal handling for graceful shutdown and artifact upload
- **Strict output directory guards** (preventing reuse of dirty directories)
- **Robust multi-process storage** (unique per-PID folders)
