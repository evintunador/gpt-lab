import os
import datetime
import logging
import json

from gpt_lab.daemon_hooks.base import BaseDaemonHook


logger = logging.getLogger(__name__)


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
        logger.info(f"FileDaemonHook initialized. "
                    f"Watching directory: {self.run_artifacts_dir}. "
                    f"Sharing metadata to: {self.watch_dir}.")

    def on_run_start(self):
        """Creates a unique JSON file with run information."""
        with open(self.watch_filepath, 'w') as f:
            json.dump({"timestamp": datetime.datetime.now().strftime("%Y%m%d%H%M%S")}, f)
        logger.info(f"Daemon hook: Created watch file at {self.watch_filepath}")

    def on_run_end(self):
        """Deletes the watch file to signal a clean exit."""
        if self.watch_filepath and os.path.exists(self.watch_filepath):
            try:
                os.remove(self.watch_filepath)
                logger.info(f"Daemon hook: Removed watch file {self.watch_filepath}")
            except OSError as e:
                logger.error(f"Daemon hook: Error removing watch file {self.watch_filepath}: {e}")
        self.watch_filepath = None