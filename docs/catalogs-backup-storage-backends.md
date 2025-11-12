# Backup Storage Backends Catalog

Backup storage backends persist completed run artifacts from `ReproducibilityManager` to a secondary location (e.g., local mirror, NFS share, S3).

## Overview

- Namespace: `gpt_lab.backup_storage_backends`
- Purpose: Durable, restorable storage of experiment outputs after runs complete or are interrupted
- Integration point: `ReproducibilityManager(..., backup_storage_backend=YourBackend(), ...)`

## Interface

Implement the base interface:

```python
from gpt_lab.backup_storage_backends.base import BaseBackupStorageBackend

class BaseBackupStorageBackend:
    def upload(self, local_source_dir: str, experiment_id: str) -> None: ...
    def download(self, experiment_id: str, local_destination_dir: str) -> None: ...
```

- `local_source_dir`: Absolute path of the run's output directory
- `experiment_id`: A stable identifier (RM uses a path relative to CWD when possible)
- `local_destination_dir`: Target directory to restore into

## Example: Local Filesystem Backend

```python
import os
import shutil
from gpt_lab.backup_storage_backends.base import BaseBackupStorageBackend

class LocalFileSystemBackend(BaseBackupStorageBackend):
    def __init__(self, remote_dir: str):
        self.remote_dir = os.path.abspath(remote_dir)
        os.makedirs(self.remote_dir, exist_ok=True)

    def upload(self, source_dir: str):
        shutil.copytree(source_dir, self.remote_dir)

    def download(self, destination_dir: str):
        shutil.copytree(self.remote_dir, destination_dir)
```

## Usage with ReproducibilityManager

In this example, the `LocalFileSystemBackend`'s `.upload()` method will be called by the `__exit__()` method of the `ReproducibilityManager`, ensuring that an experiment's artifacts are always backed up upon its completion. 

```python
from gpt_lab.reproducibility import ReproducibilityManager
from gpt_lab.backup_storage_backends.local_filesystem import LocalFileSystemBackend # part of "Core" pack

with ReproducibilityManager(
    output_dir="./runs/my_run",
    is_main_process=True,
    backup_storage_backend=LocalFileSystemBackend(remote_dir="./experiment_artifacts")
) as repro:
    train_model()
```


## Contributing

Place catalog items under:
- `catalogs/core/gpt_lab/backup_storage_backends/`
- `catalogs/packs/<pack>/gpt_lab/backup_storage_backends/`
- `experiments/<exp>/gpt_lab/backup_storage_backends/`

Backends should be idempotent and resilient to partial uploads; avoid raising unhandled exceptions during `upload`/`download`. 

