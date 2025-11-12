from abc import ABC, abstractmethod


class BaseDaemonHook(ABC):
    """
    Abstract Base Class for a daemon hook.
    This defines the interface for external processes to monitor experiment runs.
    """

    @abstractmethod
    def on_run_start(self):
        """Called when the experiment run starts."""
        raise NotImplementedError

    @abstractmethod
    def on_run_end(self):
        """Called when the experiment run ends (successfully or not)."""
        raise NotImplementedError