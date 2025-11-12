# Reproducibility

The `gpt_lab.reproducibility` module captures experiment state (git, environment) and helps persist artifacts reliably with optional storage backends and graceful shutdown.

## Overview

The reproducibility component is designed to:
- Capture complete git state (commit hash, branch, remote URL, dirty status, and GitHub URL when possible)
- Save patches of uncommitted and untracked changes for dirty repositories
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
    daemon_hook=None              # optional catalog item
) as repro:
    # repro.output_dir: absolute path you provided
    # repro.git_info: dict with git metadata
    # Your experiment code here
    train_model()
```

**Parameters:**
- `output_dir`: Root directory for experiment outputs (created only by the main process)
- `backup_storage_backend`: Optional backup storage backend for artifacts (uploads on exit when provided). Items come from the `gpt_lab.backup_storage_backends` catalog.
- `daemon_hook`: Optional hook for liveness monitoring. Items come from the `gpt_lab.daemon_hooks` catalog.

**Behavior:**

On **entry** (`__enter__`):
1. Captures git state (commit, branch, remote URL, GitHub URL when derivable)
2. Checks if working directory is dirty
3. Creates the provided output directory (main process only; determined via `gpt_lab.distributed.is_main_process()`)
4. Saves `git_info.json` with metadata
5. If dirty, creates `uncommitted_changes.patch` file
6. Calls `on_run_start` on the provided `daemon_hook` if provided.
7. Registers signal handlers for `SIGINT` (Ctrl+C) and `SIGTERM` to enable graceful shutdown.

On **exit** (`__exit__`):
1. Calls `on_run_end` on the provided `daemon_hook` if provided.
2. Uploads artifacts to the backup storage backend if provided
3. Works even if experiment exits with an error or is interrupted
4. Restores original signal handlers to avoid side effects
5. Synchronizes processes using `gpt_lab.distributed.barrier()` so non-main processes wait for the main process to finish uploads (no-op in single-process mode)

**Created Directory Structure:**
```
experiments/my_exp/runs/<specific_run_id>/
├── git_info.json
├── uncommitted_changes.patch  (if dirty)
├── checkpoints/
├── logs/
└── ... (your experiment outputs)
```

### `git_info` Dictionary

The captured git information includes:

```python
{
    "commit_hash": "a1b2c3d4e5f6...",
    "branch": "main",
    "remote_url": "git@github.com:user/repo.git",
    "github_url": "https://github.com/user/repo/commit/a1b2c3d...",
    "git_is_dirty": False
}
```

### System and Package Information

```python
from gpt_lab.reproducibility import get_system_info

info = get_system_info(git_info=repro.git_info)
# {
#   "python_version": "...",
#   "torch_version": "...",
#   "cuda_available": ...,
#   "device_count": ...,
#   "devices": [...],
#   "package_versions": {...},
#   "git_info": {...}  # if provided
# }
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

### Git State Capture

```python
from gpt_lab.reproducibility import (
    get_git_commit_hash,
    get_git_remote_url,
    get_git_branch,
    is_git_dirty,
    create_git_patch
)

# Get individual git information
commit = get_git_commit_hash()      # "a1b2c3d4e5f..."
remote = get_git_remote_url()       # "git@github.com:user/repo.git"
branch = get_git_branch()           # "main"
dirty = is_git_dirty()              # True/False

# Create patch of uncommitted and untracked changes
patch = create_git_patch()          # Full git diff as string
```

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

### Distributed Training Setup

```python
from gpt_lab.distributed import DistributedManager
from gpt_lab.reproducibility import ReproducibilityManager

with DistributedManager() as dist:
    with ReproducibilityManager(output_dir="./runs") as repro:
        # Broadcast the output directory to all ranks so they can write under the same run dir
        repro.output_dir = dist.broadcast_object(repro.output_dir, src=0)
        
        # Only the main process (rank 0) creates directories, saves git info, and uploads
        # Non-main processes proceed without those side effects
        train_distributed(...)
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