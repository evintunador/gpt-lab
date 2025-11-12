# Distributed

The `gpt_lab.distributed` module provides a unified interface for distributed training that automatically detects and configures `torchrun`, SLURM, or single-process environments. 
The hope is that eventually, as it develops, you'll be able write scripts that are completely agnostic to whether they're being run on single GPU, torchrun, or slurm without having to manage separate utilities. The module also exposes backend-agnostic, module-level accessors (rank, world size, barrier) for convenient read-only queries anywhere.

## Overview

The distributed component is designed to:
- Auto-detect distributed training environments (torchrun, SLURM)
- Initialize and manage PyTorch distributed process groups
- Provide rank-aware device placement
- Offer convenient wrappers for collective operations
- Expose module-level accessors for distributed state
- Support rank-aware deterministic seeding
- Work seamlessly in both distributed and single-process modes

## Key Component

### `DistributedManager`

A context manager that handles all aspects of distributed training setup:

```python
from gpt_lab.distributed import DistributedManager

with DistributedManager() as dist:
    # dist.rank: global rank of this process
    # dist.local_rank: local rank on this node
    # dist.world_size: total number of processes
    # dist.device: device for this process
    # dist.is_main_process: True if rank 0
    # dist.is_distributed: True if multi-process
    
    model = MyModel().to(dist.device)
    
    if dist.is_main_process:
        print(f"Training on {dist.world_size} processes")
```

When using `as dist`, the manager also mirrors common accessors as static methods, emulating a torch.distributed-like interface:

```python
with DistributedManager() as dist:
    # Static accessors (equivalent module-level functions also exist)
    dist.is_available()      # reflects torch.distributed.is_available()
    dist.is_initialized()    # True if process group initialized
    dist.get_rank()          # 0 if not initialized
    dist.get_local_rank()    # 0 if not initialized
    dist.get_world_size()    # 1 if not initialized
    dist.is_main()           # True if rank == 0
    dist.barrier()           # no-op if not initialized
```

**Attributes:**
- `rank`: Global rank (0 to world_size-1)
- `local_rank`: Local rank on this node (0 to local_world_size-1)
- `world_size`: Total number of processes
- `device`: Device for this process (`cuda:{local_rank}`, `mps`, or `cpu`)
- `is_main_process`: `True` if `rank == 0`
- `is_distributed`: `True` if running in distributed mode

**Methods:**
- `barrier()`: Synchronize all processes
- `cleanup()`: Destroy process group
- `all_gather_object(obj)`: Gather objects from all processes
- `broadcast_object(obj, src=0)`: Broadcast object from source rank
- `all_reduce(tensor, op)`: Reduce tensor across all processes
- `broadcast(tensor, src=0)`: Broadcast tensor from source rank
- `set_seed(seed)`: Set rank-aware deterministic seed

### Module-Level Accessors

These functions are available directly from `gpt_lab.distributed` and provide read-only access to the distributed state. They are backend-agnostic and safe in both single-process and distributed contexts.

```python
from gpt_lab.distributed import (
    is_available,        # reflects torch.distributed.is_available()
    is_initialized,      # True after process group init
    get_rank,            # 0 if not initialized
    get_local_rank,      # 0 if not initialized
    get_world_size,      # 1 if not initialized
    is_main_process,     # True if rank == 0
    barrier              # no-op if not initialized
)
```

Notes:
- `is_available()` reports PyTorch build capability (delegates to `torch.distributed.is_available()`).
- `is_initialized()` indicates whether a process group has been successfully created in this process.
- `barrier()` is a no-op when not initialized.

## Environment Detection

The manager automatically detects:

### 1. torchrun (PyTorch Distributed Launch)

Detected via environment variables:
- `RANK`: Global rank
- `LOCAL_RANK`: Local rank
- `WORLD_SIZE`: Total processes

```bash
# Launch with torchrun
torchrun --nproc_per_node=4 train.py
```

### 2. SLURM

Detected via environment variables:
- `SLURM_PROCID`: Global rank
- `SLURM_LOCALID`: Local rank
- `SLURM_NTASKS`: Total processes

```bash
# Launch with SLURM
srun --nodes=2 --ntasks-per-node=4 --gpus-per-node=4 python train.py
```

### 3. Single Process

If no distributed environment is detected, runs in single-process mode:
- `rank = 0`
- `world_size = 1`
- `is_distributed = False`

## Usage Examples

### Basic Distributed Training

```python
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from gpt_lab.distributed import DistributedManager

def train():
    with DistributedManager() as dist:
        # Create model and move to device
        model = MyModel().to(dist.device)
        
        # Wrap with DDP if distributed
        if dist.is_distributed:
            model = DDP(model, device_ids=[dist.local_rank])
        
            # Create distributed sampler
            train_sampler = DistributedSampler(
                train_dataset,
                num_replicas=dist.world_size,
                rank=dist.rank,
                shuffle=True
            )
        else:
            train_sampler = RandomSampler(
                train_dataset,
                shuffle=True
            )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=32,
            sampler=train_sampler
        )
        
        optimizer = torch.optim.AdamW(model.parameters())
        
        for epoch in range(num_epochs):
            # Set epoch for sampler to ensure different shuffling each epoch
            if dist.is_distributed:
                train_sampler.set_epoch(epoch)
            
            for batch in train_loader:
                batch = {k: v.to(dist.device) for k, v in batch.items()}
                
                loss = model(**batch).loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            
            # Synchronize before evaluation
            dist.barrier()
            
            # Only main process logs/saves
            if dist.is_main_process:
                print(f"Epoch {epoch} complete")

if __name__ == "__main__":
    train()
```

### Collective Operations

#### All Gather

Gather objects or tensors from all processes:

```python
with DistributedManager() as dist:
    # Gather objects
    local_metrics = {"rank": dist.rank, "loss": 0.5 - dist.rank * 0.1}
    all_metrics = dist.all_gather_object(local_metrics)
    
    if dist.is_main_process:
        print("Metrics from all processes:", all_metrics)
        # [{"rank": 0, "loss": 0.5}, {"rank": 1, "loss": 0.4}, ...]
```

#### Broadcast

Broadcast object or tensor from one process to all others:

```python
with DistributedManager() as dist:
    # Broadcast configuration from main process
    if dist.is_main_process:
        config = {"lr": 1e-4, "batch_size": 32}
    else:
        config = None
    
    # All processes receive the config from rank 0
    config = dist.broadcast_object(config, src=0)
    print(f"Rank {dist.rank} received config: {config}")
```

#### All Reduce

Reduce tensors across all processes:

```python
import torch.distributed as torch_dist
from gpt_lab.distributed import DistributedManager

with DistributedManager() as dist:
    # Sum loss across all processes
    loss = torch.tensor(1.0).to(dist.device)
    total_loss = dist.all_reduce(loss, op=torch_dist.ReduceOp.SUM)
    
    # Average loss
    avg_loss = total_loss / dist.world_size
    
    if dist.is_main_process:
        print(f"Average loss: {avg_loss.item()}")
```

#### Barrier

Synchronize all processes:

```python
from gpt_lab.distributed import DistributedManager, barrier

with DistributedManager() as dist:
    # Each process does some work
    process_data()
    
    # Wait for all processes to finish
    dist.barrier()     # or barrier()
    
    # Only main process saves results
    if dist.is_main_process:
        save_results()
    
    # Wait for save to complete
    barrier()          # module-level barrier is equivalent
```

### Rank-Aware Seeding

Set deterministic seeds that differ across ranks:

```python
from gpt_lab.distributed import DistributedManager

with DistributedManager() as dist:
    # Each rank gets a different but deterministic seed
    dist.set_seed(42)
    # Rank 0 gets seed 42
    # Rank 1 gets seed 43
    # Rank 2 gets seed 44
    # ...
    
    # This ensures:
    # 1. Each process has different randomness (important for data augmentation)
    # 2. Results are reproducible across runs
    # 3. Same experiment run gives same results
```

## Testing

Comprehensive tests are in `tests/src/gpt_lab/test_distributed.py`:

```bash
pytest tests/src/gpt_lab/test_distributed.py -v
```

**Test coverage includes:**
- Basic context manager functionality
- Device selection logic
- CPU fallback behavior
- Single-process collective operations
- Rank-aware seeding
- Object gathering and broadcasting

## Contributing

To contribute to the distributed component:

1. **Adding Features**: Extend functionality in `src/gpt_lab/distributed.py`
   - Add support for other distributed backends (Horovod, DeepSpeed)
   - Add gradient clipping utilities for distributed training
   - Add distributed profiling utilities
   - Add FSDP (Fully Sharded Data Parallel) support

2. **Adding Tests**: Add test cases to `src/gpt_lab/tests/test_distributed.py`
   - Test with mock distributed environments
   - Test collective operations
   - Test error handling
   - Add integration tests with actual multi-process setup

3. **Guidelines**:
   - Maintain backward compatibility
   - Support both distributed and single-process modes
   - Handle edge cases gracefully
   - Document environment variable requirements
