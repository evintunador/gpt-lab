import os
import datetime
import uuid
import logging
import json
from typing import Optional, Dict, Any

from gpt_lab.reproducibility.daemon_hooks.base import BaseDaemonHook


logger = logging.getLogger(__name__)


class FileDaemonHook(BaseDaemonHook):
    """
    A daemon hook that creates a watch file on run start and deletes it on run end.
    An external daemon can monitor the watch directory for these files.
    """

    def __init__(self, watch_dir: str = ".watch_runs"):
        self.watch_dir = os.path.abspath(watch_dir)
        os.makedirs(self.watch_dir, exist_ok=True)
        self.watch_file_path: Optional[str] = None
        logger.info(f"[DaemonHook] FileDaemonHook initialized. Watching directory: {self.watch_dir}")

    def on_run_start(self, run_info: Dict[str, Any]):
        """Creates a unique JSON file with run information."""
        unique_id = uuid.uuid4()
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"{timestamp}_{run_info.get('pid', 'unknown_pid')}_{unique_id}.json"
        self.watch_file_path = os.path.join(self.watch_dir, filename)
        
        with open(self.watch_file_path, 'w') as f:
            json.dump(run_info, f, indent=2)
        
        logger.info(f"Daemon hook: Created watch file at {self.watch_file_path}")

    def on_run_end(self):
        """Deletes the watch file to signal a clean exit."""
        if self.watch_file_path and os.path.exists(self.watch_file_path):
            try:
                os.remove(self.watch_file_path)
                logger.info(f"Daemon hook: Removed watch file {self.watch_file_path}")
            except OSError as e:
                logger.error(f"Daemon hook: Error removing watch file {self.watch_file_path}: {e}")
        self.watch_file_path = None