import os
import shutil
from typing import Optional
import logging

from gpt_lab.backup_storage_backends.base import BaseBackupStorageBackend


logger = logging.getLogger(__name__)


class LocalFileSystemBackend(BaseBackupStorageBackend):
    """A backend that saves artifacts to another local directory."""
    def __init__(
        self, 
        remote_dir: str,
        upload_ignore_patterns: Optional[list[str]] = None, 
        download_ignore_patterns: Optional[list[str]] = None
    ):
        self.remote_dir = os.path.abspath(remote_dir)
        os.makedirs(self.remote_dir, exist_ok=True)
        self.upload_ignore_patterns = upload_ignore_patterns
        self.download_ignore_patterns = download_ignore_patterns

    def upload(self, source_dir: str):
        source_dir = os.path.abspath(source_dir)
        if source_dir == self.remote_dir:
            logger.info(f"Artifacts are already in their final destination: {self.remote_dir}")
            return
        if not os.path.exists(source_dir):
            raise FileNotFoundError(f"No artifacts found at source: {source_dir}")
        ignore = shutil.ignore_patterns(*self.upload_ignore_patterns) if self.upload_ignore_patterns else None
        shutil.copytree(source_dir, self.remote_dir, ignore=ignore, dirs_exist_ok=True)
        logger.info(f"Artifacts from '{source_dir}' saved to {self.remote_dir}")

    def download(self, destination_dir: str):
        destination_dir = os.path.abspath(destination_dir)
        if self.remote_dir == destination_dir:
            logger.info(f"Artifacts are already in their final destination: {destination_dir}")
            return
        os.makedirs(destination_dir, exist_ok=True) 
        ignore = shutil.ignore_patterns(*self.download_ignore_patterns) if self.download_ignore_patterns else None
        shutil.copytree(self.remote_dir, destination_dir, ignore=ignore, dirs_exist_ok=False)
        logger.info(f"Artifacts from '{self.remote_dir}' downloaded to {destination_dir}")
