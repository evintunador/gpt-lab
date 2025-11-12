import os
from unittest import mock

import pytest
import torch

from gpt_lab.distributed import (
    DistributedManager,
    is_available,
    is_initialized,
    get_rank,
    get_local_rank,
    get_world_size,
    is_main_process,
    barrier
)


def test_standalone_functions_default_state():
    """Test that standalone functions return correct defaults before manager is used."""
    # Availability reflects torch's build capability
    assert is_available() == torch.distributed.is_available()
    # Not initialized by default; defaults for ranks/size
    assert not is_initialized()
    assert get_rank() == 0
    assert get_local_rank() == 0
    assert get_world_size() == 1
    assert is_main_process()
    barrier()  # Should be a no-op


def test_context_manager_basic():
    """Test basic context manager functionality in non-distributed environment."""
    with DistributedManager() as manager:
        assert manager is not None
        assert isinstance(manager.device, torch.device)
        assert manager.rank == 0
        assert manager.local_rank == 0
        assert manager.world_size == 1
        assert not manager.is_distributed
        assert manager.is_main_process
        
        # Test standalone functions reflect manager state
        assert is_available() == torch.distributed.is_available()
        assert is_initialized() == manager.is_initialized()
        assert get_rank() == manager.rank
        assert get_local_rank() == manager.local_rank
        assert get_world_size() == manager.world_size
        assert is_main_process() == manager.is_main_process

    # Test that state is reset after exit
    assert not is_initialized()
    assert get_rank() == 0
    assert get_world_size() == 1


def test_device_selection_logic():
    """Test device selection priority: CUDA > MPS > CPU."""
    with DistributedManager() as manager:
        # Device should be selected based on availability
        if torch.cuda.is_available():
            assert manager.device.type == "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            assert manager.device.type == "mps"
        else:
            assert manager.device.type == "cpu"


@mock.patch('torch.cuda.is_available', return_value=False)
@mock.patch('torch.backends.mps.is_available', return_value=False)
def test_cpu_fallback(mock_mps, mock_cuda):
    """Test that device falls back to CPU when no accelerators available."""
    with DistributedManager() as manager:
        assert manager.device.type == "cpu"


def test_all_gather_object_single_process():
    """Test all_gather_object in single process mode."""
    with DistributedManager() as manager:
        test_obj = {"key": "value", "number": 42}
        result = manager.all_gather_object(test_obj)
        assert result == [test_obj]
        assert len(result) == 1


def test_all_reduce_single_process():
    """Test all_reduce in single process mode."""
    with DistributedManager() as manager:
        tensor = torch.tensor([1.0, 2.0, 3.0])
        original = tensor.clone()
        result = manager.all_reduce(tensor)
        # In single process mode, tensor should be unchanged
        assert torch.allclose(result, original)
        assert torch.allclose(tensor, original)


def test_broadcast_single_process():
    """Test broadcast in single process mode."""
    with DistributedManager() as manager:
        tensor = torch.tensor([5.0, 10.0, 15.0])
        original = tensor.clone()
        result = manager.broadcast(tensor, src=0)
        # In single process mode, tensor should be unchanged
        assert torch.allclose(result, original)
        assert torch.allclose(tensor, original)


def test_set_seed_deterministic():
    """Test that set_seed produces deterministic behavior."""
    with DistributedManager() as manager:
        # Test that setting the same seed produces the same random numbers
        manager.set_seed(42)
        rand1 = torch.rand(3)
        
        manager.set_seed(42)
        rand2 = torch.rand(3)
        
        assert torch.allclose(rand1, rand2)


def test_set_seed_rank_aware():
    """Test that set_seed is rank-aware (different seeds for different ranks)."""
    # Test by directly modifying rank on shared state via separate contexts
    with DistributedManager() as manager1:
        # emulate rank 0
        manager1.set_seed(42)
        rand1 = torch.rand(3)
    
    with DistributedManager() as manager2:
        # emulate rank 1 by temporarily tweaking internal state
        from gpt_lab import distributed as dist_module
        dist_module._DIST_STATE["rank"] = 1
        try:
            manager2.set_seed(42)
            rand2 = torch.rand(3)
        finally:
            dist_module._DIST_STATE["rank"] = 0
    
    # Different ranks should produce different random numbers even with same base seed
    assert not torch.allclose(rand1, rand2)


@mock.patch.dict(os.environ, {"WORLD_SIZE": "2", "RANK": "0", "LOCAL_RANK": "0"}, clear=True)
@mock.patch('torch.distributed.is_available', return_value=True)
@mock.patch('torch.distributed.init_process_group')
@mock.patch('torch.distributed.destroy_process_group')
@mock.patch('torch.cuda.is_available', return_value=True)
@mock.patch('torch.cuda.set_device')
def test_torchrun_environment_detection(mock_set_device, mock_cuda_available, mock_destroy, mock_init, mock_available):
    """Test detection of torchrun environment variables."""
    with DistributedManager() as manager:
        # Should detect distributed environment
        mock_init.assert_called_once()
        assert manager.is_distributed
        assert manager.rank == 0
        assert manager.local_rank == 0
        assert manager.world_size == 2
        
        # Test standalone functions
        assert is_initialized()
        assert get_rank() == 0
        assert get_local_rank() == 0
        assert get_world_size() == 2
        assert is_main_process()
    
    mock_destroy.assert_called_once()

    # Test that state is reset after exit
    assert not is_initialized()
    assert get_rank() == 0
    assert get_world_size() == 1


@mock.patch.dict(os.environ, {
    "SLURM_PROCID": "1", 
    "SLURM_LOCALID": "1", 
    "SLURM_NTASKS": "4",
    "SLURM_SRUN_COMM_HOST": "node001",
    "SLURM_SRUN_COMM_PORT": "12345"
}, clear=True)
@mock.patch('torch.distributed.is_available', return_value=True)
@mock.patch('torch.distributed.init_process_group')
@mock.patch('torch.distributed.destroy_process_group')
@mock.patch('torch.cuda.is_available', return_value=True)
@mock.patch('torch.cuda.set_device')
def test_slurm_environment_detection(mock_set_device, mock_cuda_available, mock_destroy, mock_init, mock_available):
    """Test detection of SLURM environment variables."""
    with DistributedManager() as manager:
        mock_init.assert_called_once()
        assert manager.is_distributed
        assert manager.rank == 1
        assert manager.local_rank == 1
        assert manager.world_size == 4
        
        # Test standalone functions
        assert is_initialized()
        assert get_rank() == 1
        assert get_local_rank() == 1
        assert get_world_size() == 4
        assert not is_main_process()

        # Check that SLURM sets master address/port
        assert os.environ["MASTER_ADDR"] == "node001"
        assert os.environ["MASTER_PORT"] == "12345"
    
    mock_destroy.assert_called_once()
