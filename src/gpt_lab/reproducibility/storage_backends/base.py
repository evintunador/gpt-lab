from abc import ABC, abstractmethod
from typing import Optional


class BaseStorageBackend(ABC):
    """
    Abstract Base Class for an artifact storage backend.
    This defines the interface that all storage backends must implement.
    """
    @abstractmethod
    def __init__(self, remote_dir: str):
        raise NotImplementedError

    @abstractmethod
    def upload(self, destination_dir: str):
        """Uploads artifacts from a source to a destination directory."""
        raise NotImplementedError

    @abstractmethod
    def download(self, source_dir: str):
        """Downloads artifacts from a source to a destination directory."""
        raise NotImplementedError