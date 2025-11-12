from abc import ABC, abstractmethod


class BaseBackupStorageBackend(ABC):
    """
    Abstract Base Class for an artifact storage backend.
    This defines the interface that all storage backends must implement.
    """
    @abstractmethod
    def __init__(self, remote_dir: str):
        raise NotImplementedError

    @abstractmethod
    def upload(self, source_dir: str):
        """Uploads artifacts from a source to a remote destination directory."""
        raise NotImplementedError

    @abstractmethod
    def download(self, destination_dir: str):
        """Downloads artifacts from a remote source to a local destination directory."""
        raise NotImplementedError