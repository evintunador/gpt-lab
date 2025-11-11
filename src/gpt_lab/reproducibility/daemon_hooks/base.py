from abc import ABC, abstractmethod
from typing import Dict, Any


class BaseDaemonHook(ABC):
    """
    Abstract Base Class for a daemon hook.
    This defines the interface for external processes to monitor experiment runs.
    """

    @abstractmethod
    def on_run_start(self, run_info: Dict[str, Any]):
        """Called when the experiment run starts."""
        pass

    @abstractmethod
    def on_run_end(self):
        """Called when the experiment run ends (successfully or not)."""
        pass