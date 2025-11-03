# Reproducibility

The `gpt_lab.reproducibility` module provides comprehensive experiment reproducibility by capturing git state, creating timestamped output directories, and enabling exact experiment restoration.

## Overview

The reproducibility component is designed to:
- Capture complete git state (commit hash, branch, remote URL, dirty status)
- Save patches of uncommitted changes for dirty repositories
- Create unique timestamped directories for each experiment run
- Store artifacts with customizable storage backends
- Enable exact restoration of experiment code state

## Key Components

### `ReproducibilityManager`

A context manager that captures experiment state and manages artifact storage:

```python
from gpt_lab.reproducibility import ReproducibilityManager

with ReproducibilityManager(
    output_dir="./experiments/my_exp/runs",
    storage_backend=None,  # Uses LocalFileSystemBackend by default
    is_main_process=True
) as repro:
    # repro.output_dir: unique timestamped directory
    # repro.git_info: dict with git metadata
    # Your experiment code here
    train_model()
```

**Parameters:**
- `output_dir`: Root directory for experiment outputs
- `storage_backend`: Secondary/backup storage backend for artifacts (default: `LocalFileSystemBackend` is equivalent to not having a secondary/backup storage location)
- `is_main_process`: Whether this is the main process (for distributed training)
- `daemon_hook`: An optional hook for external monitoring of the run's liveness.

**Behavior:**

On **entry** (`__enter__`):
1. Captures git state (commit, branch, remote URL)
2. Checks if working directory is dirty
3. Creates timestamped output directory: `{timestamp}_{commit_short}/`
4. Saves `git_info.json` with metadata
5. If dirty, creates `uncommitted_changes.patch` file
6. Calls `on_run_start` on the provided `daemon_hook` if provided.
7. Registers signal handlers for `SIGINT` (Ctrl+C) and `SIGTERM` to enable graceful shutdown.

On **exit** (`__exit__`):
1. Calls `on_run_end` on the provided `daemon_hook` if provided.
2. Uploads artifacts to secondary/backup storage backend (saving to local designated folder was already being done live during the experiment)
3. Works even if experiment exits with an error or is interrupted
4. Restores original signal handlers to avoid side effects.

**Created Directory Structure:**
```
experiments/my_exp/runs/
└── 2025-10-12_14-30-45_a1b2c3d/
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
    "was_dirty": False
}
```

### Storage Backends

#### `LocalFileSystemBackend`

Default backend that copies artifacts to another local directory:

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
from gpt_lab.reproducibility import BaseStorageBackend

class S3StorageBackend(BaseStorageBackend):
    def upload(self, local_source_dir: str, experiment_id: str):
        # Upload to S3
        pass
    
    def download(self, experiment_id: str, local_destination_dir: str):
        # Download from S3
        pass
```

### `restore_experiment_state()`

WIP

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

# Create patch of uncommitted changes
patch = create_git_patch()          # Full git diff as string
```

## Usage Examples

### Basic Experiment Setup

```python
import logging
from gpt_lab.reproducibility import ReproducibilityManager
from gpt_lab.logger import setup_experiment_logging, get_system_info

with ReproducibilityManager(output_dir="./runs") as repro:
    # Setup logging
    log_dir = os.path.join(repro.output_dir, "logs")
    setup_experiment_logging(log_dir, rank=0, is_main_process=True)
    
    logger = logging.getLogger(__name__)
    
    # Log system info with git info
    system_info = get_system_info(git_info=repro.git_info)
    logger.info("Experiment started", extra=system_info)
    
    # Log experiment directory
    logger.info(f"Output directory: {repro.output_dir}")
    
    # Run experiment
    train_model()
```

### Distributed Training Setup

```python
from gpt_lab.reproducibility import ReproducibilityManager
from gpt_lab.distributed import DistributedManager

dist = DistributedManager()

# Only main process captures git state and creates directories
with ReproducibilityManager(
    output_dir="./runs",
    is_main_process=dist.is_main_process
) as repro:
    # Main process has repro.output_dir set
    # Worker processes have repro.output_dir = None
    
    if dist.is_main_process:
        print(f"Output directory: {repro.output_dir}")
    
    # Broadcast output directory to all processes
    if dist.is_main_process:
        output_dir = repro.output_dir
    else:
        output_dir = None
    
    output_dir = dist.broadcast_object(output_dir, src=0)
    
    # All processes can now use the same output directory
    train_distributed(output_dir, dist)
```

### Custom Storage Backend (S3 Example)

```python
import boto3
from gpt_lab.reproducibility import BaseStorageBackend, ReproducibilityManager

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
    storage_backend=s3_backend
) as repro:
    train_model()
    # Artifacts automatically uploaded to S3 on exit
```

### Accessing Git Info in Experiments

```python
from gpt_lab.reproducibility import ReproducibilityManager
from gpt_lab.checkpointer import save_checkpoint

with ReproducibilityManager(output_dir="./runs") as repro:
    model = MyModel()
    optimizer = torch.optim.AdamW(model.parameters())
    
    for epoch in range(num_epochs):
        train_epoch(model, optimizer)
        
        # Include git info in checkpoint metadata
        save_checkpoint(
            save_dir=os.path.join(repro.output_dir, "checkpoints"),
            filename=f"epoch_{epoch:03d}.pt",
            metadata={
                "epoch": epoch,
                "git_info": repro.git_info  # Critical for tracking!
            },
            model=model,
            optimizer=optimizer
        )
```

## CLI Usage

The `reproducibility.py` module can be used as a CLI script:

```bash
# Restore experiment from local storage
python -m gpt_lab.reproducibility \
    experiments/my_exp/runs/2025-10-12_14-30-45_a1b2c3d \
    --backend local \
    --storage_root ./experiment_artifacts \
    --restore_path ./restored
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

### 4. Test Restoration Periodically

Regularly verify you can restore old experiments:

```bash
# Restore an old experiment
python -m gpt_lab.reproducibility experiments/my_exp/runs/2025-10-12_14-30-45_a1b2c3d

# Verify files are correct
ls -la

# Return to current branch
git checkout main
```

## Testing

Comprehensive tests are in `src/gpt_lab/tests/test_reproducibility.py`:

```bash
pytest src/gpt_lab/tests/test_reproducibility.py -v
```

**Test coverage includes:**
- Clean repository state capture
- Dirty repository state capture (modified and untracked files)
- Patch file creation and application
- Storage backend upload/download
- Distributed training awareness
- Full restoration roundtrip
- Safety checks for dirty state during restoration

## Contributing

To contribute to the reproducibility component:

1. **Adding Features**: Extend functionality in `src/gpt_lab/reproducibility.py`
   - Add new storage backends (S3, GCS, Azure Blob, etc.)
   - Add git submodule support
   - Add experiment metadata extraction utilities
   - Add experiment comparison tools

2. **Adding Tests**: Add test cases to `src/gpt_lab/tests/test_reproducibility.py`
   - Test new storage backends with mock services
   - Test edge cases (detached HEAD, merge conflicts, etc.)
   - Test cross-platform compatibility
   - Test large file handling

3. **Guidelines**:
   - Always test with real git operations
   - Handle git errors gracefully
   - Maintain backward compatibility with existing artifacts
   - Document storage backend requirements
