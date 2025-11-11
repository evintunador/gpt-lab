from abc import ABC, abstractmethod
from typing import Optional


class BaseStorageBackend(ABC):
    """
    Abstract Base Class for an artifact storage backend.
    This defines the interface that all storage backends must implement.
    """

    @abstractmethod
    def upload(self, local_source_dir: str, ignore_patterns: Optional[list[str]] = None):
        """Uploads artifacts from a local directory to a destination."""
        pass

    @abstractmethod
    def download(self, remote_source_dir: str, local_destination_dir: str):
        """Downloads artifacts from a destination to a local directory."""
        pass