import os
import shutil
from typing import Optional
import logging

from gpt_lab.reproducibility.storage_backends.base import BaseStorageBackend


logger = logging.getLogger(__name__)


class LocalFileSystemBackend(BaseStorageBackend):
    """A default backend that saves artifacts to another local directory."""

    def __init__(self, root_dir: str, ignore_patterns: Optional[list[str]] = None):
        self.root_dir = os.path.abspath(root_dir)
        self.ignore_patterns = ignore_patterns
        os.makedirs(self.root_dir, exist_ok=True)
        logger.info(f"LocalFileSystemBackend initialized at: {self.root_dir}")

    def upload(self, local_source_dir: str, experiment_id: str):
        destination = os.path.join(self.root_dir, experiment_id)
        
        if os.path.abspath(local_source_dir) == os.path.abspath(destination):
            logger.info(f"Artifacts are already in their final destination: {destination}")
            return
            
        # Use shutil's built-in ignore_patterns utility
        ignore = shutil.ignore_patterns(*self.ignore_patterns) if self.ignore_patterns else None
        shutil.copytree(local_source_dir, destination, ignore=ignore, dirs_exist_ok=True)
        logger.info(f"Artifacts for '{experiment_id}' saved to {destination}")

    def download(self, experiment_id: str, local_destination_dir: str):
        source = os.path.join(self.root_dir, experiment_id)
        if not os.path.exists(source):
            raise FileNotFoundError(f"No artifacts found for experiment '{experiment_id}' at {source}")
        if os.path.exists(local_destination_dir):
            shutil.rmtree(local_destination_dir)
        shutil.copytree(source, local_destination_dir)
        logger.info(f"Artifacts for '{experiment_id}' downloaded to {local_destination_dir}")
