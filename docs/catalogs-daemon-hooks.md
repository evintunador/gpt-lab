# Daemon Hooks Catalog

Daemon hooks provide liveness and lifecycle integrations for experiments managed by `ReproducibilityManager`. They are optional plug-ins invoked on run start and end.

## Overview

- Namespace: `gpt_lab.daemon_hooks`
- Purpose: Receive lifecycle callbacks to integrate with external monitors, dashboards, or job controllers
- Integration point: `ReproducibilityManager(..., daemon_hook=YourHook(), ...)`

## Interface

Implement the base interface:

```python
from gpt_lab.daemon_hooks.base import BaseDaemonHook

class BaseDaemonHook:
    def on_run_start(self) -> None: ...
    def on_run_end(self) -> None: ...
```

## Example: File Watch Hook

```python
import os
import datetime
import json

from gpt_lab.daemon_hooks.base import BaseDaemonHook


class FileDaemonHook(BaseDaemonHook):
    """
    A daemon hook that creates a watch file on run start and deletes it on run end.
    An external daemon can monitor the watch directory for these files.
    """

    def __init__(self, watch_dir: str, run_artifacts_dir: str):
        self.watch_dir = os.path.abspath(watch_dir)
        os.makedirs(self.watch_dir, exist_ok=True)
        self.watch_filepath = os.path.join(self.watch_dir, f"{os.getpid()}.json")
        self.run_artifacts_dir = os.path.abspath(run_artifacts_dir)
        if not os.path.isdir(self.run_artifacts_dir):
            raise FileNotFoundError(f"Run artifacts directory does not exist: {self.run_artifacts_dir}")

    def on_run_start(self):
        """Creates a unique JSON file with run information."""
        with open(self.watch_filepath, 'w') as f:
            json.dump({"timestamp": datetime.datetime.now().strftime("%Y%m%d%H%M%S")}, f)

    def on_run_end(self):
        """Deletes the watch file to signal a clean exit."""
        if self.watch_filepath and os.path.exists(self.watch_file_path):
            os.remove(self.watch_file_path)
        self.watch_file_path = None
```

## Usage with ReproducibilityManager

```python
from gpt_lab.reproducibility import ReproducibilityManager
from gpt_lab.daemon_hooks.file_daemon_hook import FileDaemonHook  # example item path

with ReproducibilityManager(
    output_dir="./runs/my_run",
    is_main_process=True,
    daemon_hook=FileDaemonHook(watch_dir=".watch_runs", run_artifacts_dir="./runs/my_run")
) as repro:
    train_model()
```

## Contributing

Place catalog items under:
- `catalogs/core/gpt_lab/daemon_hooks/`
- `catalogs/packs/<pack>/gpt_lab/daemon_hooks/`
- `experiments/<exp>/gpt_lab/daemon_hooks/`

Keep hooks non-blocking and robust to failures; avoid raising in callbacks. 

