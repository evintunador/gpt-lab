import os
import random

import pytest
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from gpt_lab.checkpointer import save_checkpoint, load_checkpoint
from gpt_lab.device import get_default_device
from gpt_lab.reproducibility import get_git_commit_hash


class SimpleModel(nn.Module):
    """A simple model for testing purposes."""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 2)

    def forward(self, x):
        return self.linear(x)


@pytest.fixture
def distributed_env():
    """Fixture to set up a single-node distributed environment for testing."""
    if not dist.is_available():
        pytest.skip("torch.distributed is not available")

    port = random.randint(10000, 65535)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"

    backend = "gloo"
    dist.init_process_group(backend=backend)
    yield
    dist.destroy_process_group()
    del os.environ["MASTER_ADDR"]
    del os.environ["MASTER_PORT"]
    del os.environ["RANK"]
    del os.environ["WORLD_SIZE"]


def _compare_optimizer_states(sd_orig, sd_loaded, device):
    """Helper to compare two optimizer state_dicts."""
    assert len(sd_orig['param_groups']) == len(sd_loaded['param_groups']), "Number of param groups differs"
    for pg_orig, pg_loaded in zip(sd_orig['param_groups'], sd_loaded['param_groups']):
        pg_orig.pop('params', None)
        pg_loaded.pop('params', None)
        assert pg_orig == pg_loaded, "Optimizer param_groups do not match"

    assert sd_orig['state'].keys() == sd_loaded['state'].keys(), "Optimizer state keys do not match"
    for param_id in sd_orig['state']:
        state_orig = sd_orig['state'][param_id]
        state_loaded = sd_loaded['state'][param_id]
        for key in state_orig:
            val_orig = state_orig[key]
            val_loaded = state_loaded[key]
            if isinstance(val_orig, torch.Tensor):
                assert torch.equal(val_orig.to(device), val_loaded.to(device)), f"Optimizer state tensor '{key}' does not match"
            else:
                assert val_orig == val_loaded, f"Optimizer state value '{key}' does not match"


def _checkpointing_roundtrip_test(device: torch.device, tmp_path):
    """
    Core test logic for saving and loading a checkpoint.
    """
    save_dir = tmp_path
    filename = "test_checkpoint.pt"
    filepath = os.path.join(save_dir, filename)

    model_orig = SimpleModel().to(device)
    optimizer_orig = optim.Adam(model_orig.parameters(), lr=0.001)

    metadata_orig = {
        "epoch": 10,
        "step": 1234,
        "best_val_loss": 0.05,
    }

    optimizer_orig.zero_grad()
    dummy_input = torch.randn(4, 10, device=device)
    loss = model_orig(dummy_input).sum()
    loss.backward()
    optimizer_orig.step()

    save_checkpoint(
        filepath=filepath,
        metadata=metadata_orig,
        model=model_orig,
        optimizer=optimizer_orig,
    )
    assert os.path.exists(filepath), "Checkpoint file was not created"

    model_loaded = SimpleModel().to(device)
    optimizer_loaded = optim.Adam(model_loaded.parameters(), lr=0.001)

    assert not torch.equal(
        next(iter(model_orig.parameters())).data,
        next(iter(model_loaded.parameters())).data,
    ), "Models were already identical before loading"

    loaded_metadata = load_checkpoint(
        filepath=filepath,
        model=model_loaded,
        optimizer=optimizer_loaded,
    )

    assert loaded_metadata == metadata_orig, "Metadata was not loaded correctly"

    for p_orig, p_loaded in zip(model_orig.parameters(), model_loaded.parameters()):
        assert torch.equal(
            p_orig.data, p_loaded.data
        ), "Model parameters do not match after loading"

    sd_orig = optimizer_orig.state_dict()
    sd_loaded = optimizer_loaded.state_dict()
    _compare_optimizer_states(sd_orig, sd_loaded, device)


def test_checkpointing_roundtrip(tmp_path):
    """
    Pytest wrapper to run the checkpointing roundtrip test on the default device.
    """
    device = get_default_device()
    _checkpointing_roundtrip_test(device, tmp_path)


def _run_compatibility_test(
    tmp_path,
    device,
    create_save_objects,
    create_load_objects,
):
    """
    Generic test logic for saving one model type and loading into another.
    It takes functions to create the models/optimizers to avoid initialization issues.
    """
    filename = "compat_test.pt"
    filepath = os.path.join(tmp_path, filename)

    model_to_save, optimizer_to_save = create_save_objects()

    optimizer_to_save.zero_grad()
    dummy_input = torch.randn(4, 10, device=device)
    loss = model_to_save(dummy_input).sum()
    loss.backward()
    optimizer_to_save.step()

    save_checkpoint(
        filepath=filepath,
        model=model_to_save,
        optimizer=optimizer_to_save,
        metadata={},
    )

    model_to_load, optimizer_to_load = create_load_objects()

    load_checkpoint(
        filepath=filepath,
        model=model_to_load,
        optimizer=optimizer_to_load,
    )

    raw_model_saved = model_to_save
    if hasattr(raw_model_saved, "module"):
        raw_model_saved = raw_model_saved.module
    if hasattr(raw_model_saved, "_orig_mod"):
        raw_model_saved = raw_model_saved._orig_mod

    raw_model_loaded = model_to_load
    if hasattr(raw_model_loaded, "module"):
        raw_model_loaded = raw_model_loaded.module
    if hasattr(raw_model_loaded, "_orig_mod"):
        raw_model_loaded = raw_model_loaded._orig_mod

    for p_orig, p_loaded in zip(
        raw_model_saved.parameters(), raw_model_loaded.parameters()
    ):
        assert torch.equal(
            p_orig.data, p_loaded.data
        ), "Model parameters do not match"

    _compare_optimizer_states(
        optimizer_to_save.state_dict(), optimizer_to_load.state_dict(), device
    )


# --- DDP Tests ---
@pytest.mark.skipif(
    get_default_device().type == "mps", reason="DDP is not supported on MPS backend"
)
def test_ddp_save_raw_load_ddp(tmp_path, distributed_env):
    """Tests saving a raw model and loading into a DDP-wrapped one."""
    device = get_default_device()
    def create_raw():
        model = SimpleModel().to(device)
        return model, optim.Adam(model.parameters())
    def create_ddp():
        model = DDP(SimpleModel().to(device), device_ids=[device.index] if device.type == "cuda" else None)
        return model, optim.Adam(model.parameters())
    _run_compatibility_test(tmp_path, device, create_raw, create_ddp)

@pytest.mark.skipif(
    get_default_device().type == "mps", reason="DDP is not supported on MPS backend"
)
def test_ddp_save_ddp_load_raw(tmp_path, distributed_env):
    """Tests saving a DDP-wrapped model and loading into a raw one."""
    device = get_default_device()
    def create_raw():
        model = SimpleModel().to(device)
        return model, optim.Adam(model.parameters())
    def create_ddp():
        model = DDP(SimpleModel().to(device), device_ids=[device.index] if device.type == "cuda" else None)
        return model, optim.Adam(model.parameters())
    _run_compatibility_test(tmp_path, device, create_ddp, create_raw)


# --- torch.compile Tests ---
@pytest.mark.skipif(not hasattr(torch, 'compile'), reason="torch.compile not available")
def test_compile_save_raw_load_compiled(tmp_path):
    """Tests saving a raw model and loading into a compiled one."""
    device = get_default_device()
    def create_raw():
        model = SimpleModel().to(device)
        return model, optim.Adam(model.parameters())
    def create_compiled():
        model = torch.compile(SimpleModel().to(device))
        return model, optim.Adam(model.parameters())
    _run_compatibility_test(tmp_path, device, create_raw, create_compiled)

@pytest.mark.skipif(not hasattr(torch, 'compile'), reason="torch.compile not available")
def test_compile_save_compiled_load_raw(tmp_path):
    """Tests saving a compiled model and loading into a raw one."""
    device = get_default_device()
    def create_raw():
        model = SimpleModel().to(device)
        return model, optim.Adam(model.parameters())
    def create_compiled():
        model = torch.compile(SimpleModel().to(device))
        return model, optim.Adam(model.parameters())
    _run_compatibility_test(tmp_path, device, create_compiled, create_raw)


# --- DDP + torch.compile Tests ---
@pytest.mark.skipif(
    get_default_device().type == "mps", reason="DDP is not supported on MPS backend"
)
@pytest.mark.skipif(not hasattr(torch, 'compile'), reason="torch.compile not available")
def test_ddp_compile_save_wrapped_load_raw(tmp_path, distributed_env):
    """Tests saving a DDP-compiled model and loading into a raw one."""
    device = get_default_device()
    def create_raw():
        model = SimpleModel().to(device)
        return model, optim.Adam(model.parameters())
    def create_wrapped():
        model = DDP(torch.compile(SimpleModel().to(device)), device_ids=[device.index] if device.type == "cuda" else None)
        return model, optim.Adam(model.parameters())
    _run_compatibility_test(tmp_path, device, create_wrapped, create_raw)
