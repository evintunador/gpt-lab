import os
from typing import TypeVar, List, Any
import logging

import torch
import torch.distributed as dist
import random
import numpy as np

logger = logging.getLogger(__name__)


_DIST_STATE = {
    "is_initialized": False,
    "rank": 0,
    "local_rank": 0,
    "world_size": 1,
}


# --- Standalone distributed accessors ---

def is_available() -> bool:
    """Checks if torch.distributed is available in this build of PyTorch."""
    return dist.is_available()


def is_initialized() -> bool:
    """Returns True if the distributed process group has been initialized."""
    return _DIST_STATE["is_initialized"]


def get_rank() -> int:
    """Gets the rank of the current process, defaulting to 0 if not in a distributed context."""
    return _DIST_STATE["rank"]


def get_local_rank() -> int:
    """Gets the local rank of the current process, defaulting to 0 if not in a distributed context."""
    return _DIST_STATE["local_rank"]


def get_world_size() -> int:
    """Gets the total number of processes, defaulting to 1 if not in a distributed context."""
    return _DIST_STATE["world_size"]


def is_main_process() -> bool:
    """Returns True if the current process is the main one (rank 0)."""
    return get_rank() == 0


def is_main() -> bool:
    # alias for clarity if users prefer shorter name
    return is_main_process()


def barrier() -> None:
    """Synchronizes all processes. Does nothing if not initialized."""
    # Use PyTorch's actual initialization state to avoid errors in tests or
    # contexts where module-level state was mocked but no process group exists.
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


T = TypeVar('T')


class DistributedManager:
    """
    A context manager to handle distributed training environments.

    This class automatically detects and initializes the process group for
    distributed training if run in a torchrun or SLURM environment. It
    manages the device placement and provides convenience methods for
    distributed operations. It also updates a module-level state that can
    be accessed via standalone functions (e.g., `get_rank()`, `is_main_process()`).
    """

    def __init__(self):
        # Device is instance-specific; ranks are read from module state via properties
        self.device: torch.device = torch.device("cpu")

    # Properties read the single shared state to avoid duplication/drift
    @property
    def is_distributed(self) -> bool:
        return is_initialized()

    @property
    def rank(self) -> int:
        return get_rank()

    @property
    def local_rank(self) -> int:
        return get_local_rank()

    @property
    def world_size(self) -> int:
        return get_world_size()

    @property
    def is_main_process(self) -> bool:
        return is_main_process()

    # Static methods emulate torch.distributed API for convenience when using `as dist`
    @staticmethod
    def is_available() -> bool:
        return is_available()

    @staticmethod
    def is_initialized() -> bool:
        return is_initialized()

    @staticmethod
    def get_rank() -> int:
        return get_rank()

    @staticmethod
    def get_local_rank() -> int:
        return get_local_rank()

    @staticmethod
    def get_world_size() -> int:
        return get_world_size()

    @staticmethod
    def is_main() -> bool:
        # alias for clarity if users prefer shorter name
        return is_main_process()

    @staticmethod
    def barrier() -> None:
        barrier()

    def __enter__(self):
        """Initializes the distributed environment."""
        if is_available() and self._is_dist_env():
            self._init_distributed()
        else:
            logger.info("Running in single-process mode")
        
        self._set_device()
        logger.info(f"Process initialized - Rank: {self.rank}/{self.world_size}, Device: {self.device}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Cleans up the distributed environment."""
        self.cleanup()

        # Reset module state to safe defaults
        _DIST_STATE["is_initialized"] = False
        _DIST_STATE["rank"] = 0
        _DIST_STATE["local_rank"] = 0
        _DIST_STATE["world_size"] = 1

    def _is_dist_env(self) -> bool:
        """Checks if the script is running in a distributed environment."""
        return "WORLD_SIZE" in os.environ or "SLURM_NTASKS" in os.environ

    def _init_distributed(self) -> None:
        """Sets up the process group and updates shared state."""
        if "SLURM_PROCID" in os.environ:
            rank = int(os.environ["SLURM_PROCID"])
            local_rank = int(os.environ["SLURM_LOCALID"])
            world_size = int(os.environ["SLURM_NTASKS"])
            # Configure master address/port from SLURM
            os.environ["MASTER_ADDR"] = os.environ["SLURM_SRUN_COMM_HOST"]
            os.environ["MASTER_PORT"] = str(int(os.environ["SLURM_SRUN_COMM_PORT"]))
            logger.info(f"Detected SLURM environment - Rank {rank}/{world_size}")
        elif "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            rank = int(os.environ["RANK"])
            local_rank = int(os.environ["LOCAL_RANK"])
            world_size = int(os.environ["WORLD_SIZE"])
            logger.info(f"Detected torchrun environment - Rank {rank}/{world_size}")
        else:
            return  # Not a distributed environment

        backend = "nccl" if torch.cuda.is_available() else "gloo"
        logger.info(f"Initializing process group with backend: {backend}")
        dist.init_process_group(
            backend=backend,
            rank=rank,
            world_size=world_size
        )

        # Update shared state after successful init
        _DIST_STATE["is_initialized"] = True
        _DIST_STATE["rank"] = rank
        _DIST_STATE["local_rank"] = local_rank
        _DIST_STATE["world_size"] = world_size
        logger.info("Process group initialized successfully")
        
    def _set_device(self) -> None:
        """Sets the device for the current process."""
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{self.local_rank}")
            torch.cuda.set_device(self.device)
            logger.debug(f"Set CUDA device to {self.device}")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            # Note: MPS does not support distributed training.
            self.device = torch.device("mps")
            logger.debug("Set device to MPS (Apple Silicon)")
        else:
            self.device = torch.device("cpu")
            logger.debug("Set device to CPU")

    def cleanup(self) -> None:
        """Destroys the process group."""
        if self.is_distributed:
            dist.destroy_process_group()

    def all_gather_object(self, obj: T) -> List[T]:
        """Gathers a pickleable object from all processes and returns a list."""
        if not self.is_distributed:
            return [obj]
        
        output_list: List[Any] = [None for _ in range(self.world_size)]
        dist.all_gather_object(output_list, obj)
        return output_list

    def broadcast_object(self, obj: T, src: int = 0) -> T:
        """
        Broadcasts a pickleable object from a source rank to all other processes.
        
        Args:
            obj: The object to broadcast. On ranks other than src, this is a placeholder
                 and its value is ignored.
            src: The rank of the process to broadcast from.

        Returns:
            The object broadcasted from the source rank.
        """
        if not self.is_distributed:
            return obj
        
        # The object needs to be in a list for broadcast_object_list
        obj_list = [obj] if self.rank == src else [None]
        dist.broadcast_object_list(obj_list, src=src)
        return obj_list[0]

    def all_reduce(self, tensor: torch.Tensor, op: dist.ReduceOp = dist.ReduceOp.SUM) -> torch.Tensor:
        """Reduces the tensor data across all processes."""
        if self.is_distributed:
            dist.all_reduce(tensor, op=op)
        return tensor

    def broadcast(self, tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
        """Broadcasts a tensor from a source rank to all other processes."""
        if self.is_distributed:
            dist.broadcast(tensor, src=src)
        return tensor

    def set_seed(self, seed: int) -> None:
        """Sets a deterministic, rank-aware seed for reproducibility."""
        final_seed = seed + self.rank
        random.seed(final_seed)
        np.random.seed(final_seed)
        torch.manual_seed(final_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(final_seed)
        logger.info(f"Set seed to {final_seed} (base: {seed}, rank: {self.rank})")
