# Reproducibility

The `gpt_lab.reproducibility` module captures experiment state (git, environment) and helps persist artifacts reliably with optional storage backends and graceful shutdown.

## Overview

The reproducibility component is designed to:
- Capture complete git state (commit hash, branch, remote URL, dirty status, and GitHub URL when possible)
- Save patches of uncommitted and untracked changes for dirty repositories
- Create the output directory you specify (main process only)
- Optionally store artifacts with customizable storage backends
- Gracefully handle interrupts (SIGINT/SIGTERM) and still upload artifacts when configured

## Key Components

### `ReproducibilityManager`

A context manager that captures experiment state and manages artifact storage:

```python
from gpt_lab.reproducibility import ReproducibilityManager

with ReproducibilityManager(
    output_dir="./experiments/my_exp/runs",
    is_main_process=True,
    storage_backend=None,
    daemon_hook=None
) as repro:
    # repro.output_dir: absolute path you provided
    # repro.git_info: dict with git metadata
    # Your experiment code here
    train_model()
```

**Parameters:**
- `output_dir`: Root directory for experiment outputs (created if `is_main_process=True`)
- `is_main_process`: Whether this is the main process (for distributed training)
- `storage_backend`: Optional secondary/backup storage backend for artifacts (if provided, uploads on exit)
- `daemon_hook`: An optional hook for external monitoring of the run's liveness.

**Behavior:**

On **entry** (`__enter__`):
1. Captures git state (commit, branch, remote URL, GitHub URL when derivable)
2. Checks if working directory is dirty
3. Creates the provided output directory (main process only)
4. Saves `git_info.json` with metadata
5. If dirty, creates `uncommitted_changes.patch` file
6. Calls `on_run_start` on the provided `daemon_hook` if provided.
7. Registers signal handlers for `SIGINT` (Ctrl+C) and `SIGTERM` to enable graceful shutdown.

On **exit** (`__exit__`):
1. Calls `on_run_end` on the provided `daemon_hook` if provided.
2. Uploads artifacts to the storage backend if provided
3. Works even if experiment exits with an error or is interrupted
4. Restores original signal handlers to avoid side effects.

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

### Storage Backends

#### `LocalFileSystemBackend`

Example backend that copies artifacts to another local directory:

```python
from gpt_lab.reproducibility import LocalFileSystemBackend

backend = LocalFileSystemBackend(root_dir="./experiment_artifacts")
```

**Methods:**
- `upload(local_source_dir, experiment_id)`: Copies directory
- `download(experiment_id, local_destination_dir)`: Restores directory

#### `BaseStorageBackend`

Abstract base class for custom backends:

```python
from gpt_lab.reproducibility.storage_backends.base import BaseStorageBackend

class S3StorageBackend(BaseStorageBackend):
    def upload(self, local_source_dir: str, experiment_id: str):
        # Upload to S3
        pass
    
    def download(self, experiment_id: str, local_destination_dir: str):
        # Download from S3
        pass
```

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

with ReproducibilityManager(output_dir="./runs", is_main_process=True) as repro:
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
from gpt_lab.reproducibility import ReproducibilityManager
# from gpt_lab.distributed import DistributedManager

is_main_process = get_is_main_process_somehow()

# Only main process captures git state and creates directories/uploads
with ReproducibilityManager(output_dir="./runs", is_main_process=is_main_process) as repro:
    if is_main_process:
        print(f"Output directory: {repro.output_dir}")
        # broadcast repro.output_dir to workers with your distributed framework
    train_distributed(...)
```

### Custom Storage Backend (S3 Example)

```python
import boto3
from gpt_lab.reproducibility import ReproducibilityManager
from gpt_lab.reproducibility.storage_backends.base import BaseStorageBackend

class S3StorageBackend(BaseStorageBackend):
    def __init__(self, bucket_name: str, prefix: str = ""):
        self.s3_client = boto3.client('s3')
        self.bucket_name = bucket_name
        self.prefix = prefix
    
    def upload(self, local_source_dir: str, experiment_id: str):
        import os
        from pathlib import Path
        
        # Upload all files in directory
        for root, dirs, files in os.walk(local_source_dir):
            for file in files:
                local_path = os.path.join(root, file)
                relative_path = os.path.relpath(local_path, local_source_dir)
                s3_key = f"{self.prefix}/{experiment_id}/{relative_path}"
                
                self.s3_client.upload_file(local_path, self.bucket_name, s3_key)
        
        print(f"Uploaded to s3://{self.bucket_name}/{self.prefix}/{experiment_id}")
    
    def download(self, experiment_id: str, local_destination_dir: str):
        import os
        
        os.makedirs(local_destination_dir, exist_ok=True)
        
        # List and download all objects with the prefix
        prefix = f"{self.prefix}/{experiment_id}/"
        response = self.s3_client.list_objects_v2(
            Bucket=self.bucket_name,
            Prefix=prefix
        )
        
        for obj in response.get('Contents', []):
            s3_key = obj['Key']
            relative_path = s3_key[len(prefix):]
            local_path = os.path.join(local_destination_dir, relative_path)
            
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            self.s3_client.download_file(self.bucket_name, s3_key, local_path)
        
        print(f"Downloaded from s3://{self.bucket_name}/{prefix}")

# Use custom backend
s3_backend = S3StorageBackend(bucket_name="my-experiments", prefix="ml-runs")

with ReproducibilityManager(
    output_dir="./runs",
    storage_backend=s3_backend,
    is_main_process=True
) as repro:
    train_model()
    # Artifacts automatically uploaded to S3 on exit
```

## Best Practices

### 1. Always Use ReproducibilityManager

Wrap your experiment entry point:

```python
def main():
    with ReproducibilityManager(output_dir="./runs", is_main_process=True) as repro:
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

Comprehensive tests are in `tests/src/gpt_lab/reproducibility/test_reproducibility.py`:

```bash
pytest tests/src/gpt_lab/reproducibility/test_reproducibility.py -v
```

**Test coverage includes:**
- Clean repository state capture
- Dirty repository state capture (modified and untracked files)
- Patch file creation and application
- Storage backend upload/download
- Distributed training awareness
- Signal handling for graceful shutdown and artifact upload