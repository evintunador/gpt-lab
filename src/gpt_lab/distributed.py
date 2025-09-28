import os
from typing import TypeVar, List, Any

import torch
import torch.distributed as dist
import random
import numpy as np


T = TypeVar('T')

class DistributedManager:
    """
    A context manager to handle distributed training environments.

    This class automatically detects and initializes the process group for
    distributed training if run in a torchrun or SLURM environment. It
    manages the device placement and provides convenience methods for
    distributed operations.

    Usage:
        with DistributedManager() as dist_manager:
            # Your training code here
            model = MyModel().to(dist_manager.device)
            if dist_manager.is_main_process:
                print(f"Running on {dist_manager.world_size} GPUs.")
            dist_manager.barrier()
    """

    def __init__(self):
        self.is_distributed: bool = False
        self.rank: int = 0
        self.local_rank: int = 0
        self.world_size: int = 1
        self.device: torch.device = torch.device("cpu")

    @staticmethod
    def is_available() -> bool:
        """Checks if the distributed package is available."""
        return dist.is_available()

    def is_initialized(self) -> bool:
        """Returns True if the distributed process group has been initialized."""
        return self.is_distributed

    def __enter__(self):
        """Initializes the distributed environment."""
        if self.is_available() and self._is_dist_env():
            self._init_distributed()
        
        self._set_device()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Cleans up the distributed environment."""
        self.cleanup()

    def _is_dist_env(self) -> bool:
        """Checks if the script is running in a distributed environment."""
        return "WORLD_SIZE" in os.environ or "SLURM_NTASKS" in os.environ

    def _init_distributed(self):
        """Sets up the process group."""
        if "SLURM_PROCID" in os.environ:
            # SLURM environment variables
            self.rank = int(os.environ["SLURM_PROCID"])
            self.local_rank = int(os.environ["SLURM_LOCALID"])
            self.world_size = int(os.environ["SLURM_NTASKS"])
            # SLURM uses HOST:PORT for master address
            # This is a common setup, might need adjustment based on specific SLURM config
            os.environ["MASTER_ADDR"] = os.environ["SLURM_SRUN_COMM_HOST"]
            os.environ["MASTER_PORT"] = str(int(os.environ["SLURM_SRUN_COMM_PORT"]))
        elif "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            # torchrun environment variables
            self.rank = int(os.environ["RANK"])
            self.local_rank = int(os.environ["LOCAL_RANK"])
            self.world_size = int(os.environ["WORLD_SIZE"])
        else:
            return  # Not a distributed environment

        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(
            backend=backend, 
            rank=self.rank, 
            world_size=self.world_size
        )
        self.is_distributed = True
        
    def _set_device(self):
        """Sets the device for the current process."""
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{self.local_rank}")
            torch.cuda.set_device(self.device)
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            # Note: MPS does not support distributed training.
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

    @property
    def is_main_process(self) -> bool:
        """Returns True if the current process is the main one (rank 0)."""
        return self.rank == 0

    def barrier(self):
        """Synchronizes all processes."""
        if self.is_distributed:
            dist.barrier()

    def cleanup(self):
        """Destroys the process group."""
        if self.is_distributed:
            dist.destroy_process_group()
            self.is_distributed = False

    def all_gather_object(self, obj: T) -> List[T]:
        """Gathers a pickleable object from all processes and returns a list."""
        if not self.is_distributed:
            return [obj]
        
        output_list: List[Any] = [None for _ in range(self.world_size)]
        dist.all_gather_object(output_list, obj)
        return output_list

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
        # Each rank gets a different seed to avoid identical-but-distributed randomness
        final_seed = seed + self.rank
        random.seed(final_seed)
        np.random.seed(final_seed)
        torch.manual_seed(final_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(final_seed)
