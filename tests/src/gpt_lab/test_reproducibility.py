import os
import subprocess
import json
from pathlib import Path
import shutil
import pytest
import signal
from unittest.mock import MagicMock
import torch

from gpt_lab.reproducibility import (
    ReproducibilityManager,
    get_git_submodules_info,
    get_git_superprojects_info,
)
from gpt_lab.backup_storage_backends.base import BaseBackupStorageBackend


# --- Test Fixtures and Helpers ---

@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    """Creates a temporary git repository for testing."""
    repo_path = tmp_path / "test_repo"
    repo_path.mkdir()
    
    # Save current directory to restore later
    original_cwd = os.getcwd()
    
    try:
        os.chdir(repo_path)
        subprocess.run(["git", "init"], check=True, capture_output=True)
        subprocess.run(["git", "config", "user.name", "Test User"], check=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"], check=True)
        
        (repo_path / "file1.txt").write_text("initial content")
        subprocess.run(["git", "add", "file1.txt"], check=True)
        subprocess.run(["git", "commit", "-m", "Initial commit"], check=True, capture_output=True)
        
        return repo_path
    finally:
        os.chdir(original_cwd)


class MockStorageBackend(BaseBackupStorageBackend):
    """A mock storage backend that records calls for testing purposes."""
    def __init__(self, remote_dir: Path):
        self.upload_calls = []
        self.download_calls = []
        self.remote_dir = remote_dir

    def upload(self, source_dir: str):
        self.upload_calls.append(source_dir)
        # Simulate the upload by copying to a "remote" location
        if self.remote_dir.exists():
            shutil.rmtree(self.remote_dir)
        shutil.copytree(source_dir, self.remote_dir)

    def download(self, destination_dir: str):
        self.download_calls.append(destination_dir)
        # Simulate download by copying from the "remote" location
        if not self.remote_dir.exists():
            raise FileNotFoundError(f"Remote not found: {self.remote_dir}")
        shutil.copytree(self.remote_dir, destination_dir)


# --- Core Functionality Tests ---

def test_manager_clean_repo(git_repo: Path):
    """Verify manager behavior in a clean git repository."""
    original_cwd = os.getcwd()
    try:
        os.chdir(git_repo)
        runs_dir = git_repo / "experiments" / "test-exp" / "runs"
        with ReproducibilityManager(output_dir=str(runs_dir)) as manager:
            output_dir = Path(manager.output_dir)
            assert output_dir.is_dir()

            # Check git_info.json
            git_info_file = output_dir / "git_info.json"
            assert git_info_file.exists()
            with open(git_info_file, 'r') as f:
                git_info = json.load(f)
            assert not git_info["git_is_dirty"]
            # New git topology fields should be present, even if empty
            assert "submodules" in git_info
            assert "superprojects" in git_info
            assert isinstance(git_info["submodules"], list)
            assert isinstance(git_info["superprojects"], list)
        
            # Check that no patch file was created
            assert not (output_dir / "uncommitted_changes.patch").exists()
    finally:
        os.chdir(original_cwd)


@pytest.mark.parametrize("dirty_type", ["modified", "untracked"])
def test_manager_dirty_repo(git_repo: Path, dirty_type: str):
    """Verify manager behavior in a dirty git repository."""
    original_cwd = os.getcwd()
    try:
        os.chdir(git_repo)
        
        if dirty_type == "modified":
            (git_repo / "file1.txt").write_text("modified content")
        elif dirty_type == "untracked":
            (git_repo / "new_file.txt").write_text("untracked file")

        runs_dir = git_repo / "experiments" / "test-exp" / "runs"
        with ReproducibilityManager(output_dir=str(runs_dir)) as manager:
            output_dir = Path(manager.output_dir)
            assert output_dir.is_dir()

            # Check git_info.json
            git_info_file = output_dir / "git_info.json"
            assert git_info_file.exists()
            with open(git_info_file, 'r') as f:
                git_info = json.load(f)
            assert git_info["git_is_dirty"]
            # New git topology fields should be present, even if empty
            assert "submodules" in git_info
            assert "superprojects" in git_info
            assert isinstance(git_info["submodules"], list)
            assert isinstance(git_info["superprojects"], list)
        
            # Check that a patch file was created and is not empty
            patch_file = output_dir / "uncommitted_changes.patch"
            assert patch_file.exists()
            assert patch_file.read_text().strip() != ""
    finally:
        os.chdir(original_cwd)


def test_manager_distributed_awareness(git_repo: Path):
    """Verify the manager does nothing on non-main processes."""
    original_cwd = os.getcwd()
    try:
        os.chdir(git_repo)
        runs_dir = git_repo / "experiments" / "test-exp" / "runs"
        # Simulate a distributed, non-main process by tweaking shared state
        from gpt_lab import distributed as dist_module
        prev = dist_module._DIST_STATE.copy()
        try:
            dist_module._DIST_STATE.update({
                "is_initialized": True,
                "rank": 1,
                "local_rank": 1,
                "world_size": 2,
            })
            with ReproducibilityManager(
                output_dir=str(runs_dir),
            ) as manager:
                assert manager._is_main_process is False
        finally:
            dist_module._DIST_STATE.update(prev)
    finally:
        os.chdir(original_cwd)


def test_manager_storage_upload(git_repo: Path, tmp_path: Path):
    """Verify that the manager calls the storage backend's upload method."""
    original_cwd = os.getcwd()
    try:
        os.chdir(git_repo)
        mock_storage = MockStorageBackend(remote_dir=tmp_path / "remote_storage")
        
        runs_dir = git_repo / "experiments" / "test-exp" / "runs"
        with ReproducibilityManager(
            output_dir=str(runs_dir),
            backup_storage_backend=mock_storage
        ) as manager:
            # Simulate creating an output file in the experiment dir
            (Path(manager.output_dir) / "results.txt").write_text("success")

        assert len(mock_storage.upload_calls) == 1
        
        # Check that the uploaded content is correct
        assert (tmp_path / "remote_storage" / "results.txt").read_text() == "success"
    finally:
        os.chdir(original_cwd)


def test_environment_snapshots_persisted_and_exposed(git_repo: Path):
    """Verify that software and runtime environment info are saved and exposed via the manager."""
    original_cwd = os.getcwd()
    try:
        os.chdir(git_repo)
        runs_dir = git_repo / "experiments" / "test-exp" / "runs"
        with ReproducibilityManager(output_dir=str(runs_dir)) as manager:
            output_dir = Path(manager.output_dir)

            # Software environment
            software_file = output_dir / "software_environment.json"
            assert software_file.exists()
            sw = json.loads(software_file.read_text())
            assert "python_version" in sw
            assert "package_versions" in sw
            assert "torch_repro_settings" in sw

            sw_prop = manager.software_environment
            assert isinstance(sw_prop, dict)
            assert "python_version" in sw_prop
            assert "torch_repro_settings" in sw_prop

            # Runtime environment
            runtime_file = output_dir / "runtime_environment.json"
            assert runtime_file.exists()
            rt = json.loads(runtime_file.read_text())
            assert "cuda_available" in rt
            assert "device_count" in rt
            assert "distributed" in rt

            rt_prop = manager.runtime_environment
            assert isinstance(rt_prop, dict)
            assert "cuda_available" in rt_prop
            assert "distributed" in rt_prop
    finally:
        os.chdir(original_cwd)


def test_initial_rng_state_persisted_and_exposed(git_repo: Path):
    """Verify that initial RNG state is saved and exposed via the manager."""
    original_cwd = os.getcwd()
    try:
        os.chdir(git_repo)
        runs_dir = git_repo / "experiments" / "test-exp" / "runs"
        with ReproducibilityManager(output_dir=str(runs_dir)) as manager:
            output_dir = Path(manager.output_dir)

            # File was written
            rng_file = output_dir / "rng_state_initial.pt"
            assert rng_file.exists()
            # RNG state includes non-tensor Python/numpy objects; use weights_only=False
            # to avoid PyTorch's safe-loading restrictions.
            loaded = torch.load(rng_file, weights_only=False)
            assert isinstance(loaded, dict)
            for key in ("torch", "numpy", "random"):
                assert key in loaded

            # Properties/methods are available
            initial = manager.initial_rng_state
            assert isinstance(initial, dict)
            current = manager.get_rng_states()
            assert isinstance(current, dict)

            # Sanity check set_rng_states works round-trip for torch RNG
            rng_before = manager.get_rng_states()
            _ = torch.rand(1)  # change RNG
            manager.set_rng_states(rng_before)
            rng_after = manager.get_rng_states()
            assert torch.equal(rng_before["torch"], rng_after["torch"])
            # Final RNG state file should exist by the time context exits
        final_rng_file = runs_dir / "rng_state_final.pt"
        assert final_rng_file.exists()
        final_state = torch.load(final_rng_file, weights_only=False)
        assert isinstance(final_state, dict)
        for key in ("torch", "numpy", "random"):
            assert key in final_state

    finally:
        os.chdir(original_cwd)


def test_run_invocation_persisted_and_exposed(git_repo: Path):
    """Verify that run invocation information is saved and exposed."""
    original_cwd = os.getcwd()
    try:
        os.chdir(git_repo)
        runs_dir = git_repo / "experiments" / "test-exp" / "runs"
        with ReproducibilityManager(output_dir=str(runs_dir)) as manager:
            output_dir = Path(manager.output_dir)

            inv_file = output_dir / "run_invocation.json"
            assert inv_file.exists()
            data = json.loads(inv_file.read_text())
            assert "argv" in data
            assert "env" in data
            assert isinstance(data["argv"], list)
            assert isinstance(data["env"], dict)

            # Property mirrors the file contents' structure
            ri = manager.run_invocation
            assert isinstance(ri, dict)
            assert "argv" in ri and isinstance(ri["argv"], list)
            assert "env" in ri and isinstance(ri["env"], dict)
    finally:
        os.chdir(original_cwd)


def test_torch_determinism_persisted_and_exposed(git_repo: Path):
    """Verify that torch determinism metadata is saved and exposed."""
    original_cwd = os.getcwd()
    try:
        os.chdir(git_repo)
        runs_dir = git_repo / "experiments" / "test-exp" / "runs"
        with ReproducibilityManager(output_dir=str(runs_dir)) as manager:
            output_dir = Path(manager.output_dir)

            det_file = output_dir / "torch_determinism.json"
            assert det_file.exists()
            data = json.loads(det_file.read_text())
            assert isinstance(data, dict)
            # We don't assert on specific keys to avoid PyTorch-version flakiness,
            # but the snapshot should never be empty.
            assert data

            td = manager.torch_determinism
            assert isinstance(td, dict)
            assert td
    finally:
        os.chdir(original_cwd)


def test_distributed_topology_persisted_and_exposed(git_repo: Path):
    """Verify that distributed topology snapshot is saved and exposed."""
    original_cwd = os.getcwd()
    try:
        os.chdir(git_repo)
        runs_dir = git_repo / "experiments" / "test-exp" / "runs"
        with ReproducibilityManager(output_dir=str(runs_dir)) as manager:
            output_dir = Path(manager.output_dir)

            topo_file = output_dir / "distributed_topology.json"
            assert topo_file.exists()
            data = json.loads(topo_file.read_text())
            assert isinstance(data, dict)
            for key in ("is_initialized", "is_distributed", "world_size", "rank", "local_rank"):
                assert key in data

            topo = manager.distributed_topology
            assert isinstance(topo, dict)
            for key in ("is_initialized", "is_distributed", "world_size", "rank", "local_rank"):
                assert key in topo
    finally:
        os.chdir(original_cwd)


def test_daemon_hook_lifecycle(git_repo: Path):
    """Verify the DaemonHook's start and end methods are called."""
    original_cwd = os.getcwd()
    try:
        os.chdir(git_repo)
        mock_hook = MagicMock()
        
        runs_dir = git_repo / "experiments" / "test-exp" / "runs"
        
        with ReproducibilityManager(
            output_dir=str(runs_dir),
            daemon_hook=mock_hook
        ):
            # 1. Check on_run_start was called
            mock_hook.on_run_start.assert_called_once()

        # 2. Check on_run_end was called on __exit__
        mock_hook.on_run_end.assert_called_once()

    finally:
        os.chdir(original_cwd)


def test_daemon_hook_survives_exception(git_repo: Path):
    """Verify the hook's on_run_end is called even if the experiment fails."""
    original_cwd = os.getcwd()
    try:
        os.chdir(git_repo)
        mock_hook = MagicMock()
        
        runs_dir = git_repo / "experiments" / "test-exp" / "runs"
        
        with pytest.raises(ValueError, match="Experiment failed"):
            with ReproducibilityManager(
                output_dir=str(runs_dir),
                daemon_hook=mock_hook
            ):
                mock_hook.on_run_start.assert_called_once()
                raise ValueError("Experiment failed")

        # on_run_end should still be called
        mock_hook.on_run_end.assert_called_once()
    finally:
        os.chdir(original_cwd)


@pytest.mark.parametrize("sig", [signal.SIGINT, signal.SIGTERM])
def test_manager_signal_handling(git_repo: Path, tmp_path: Path, sig: int):
    """Verify that artifacts are saved on SIGINT or SIGTERM."""
    # This test might be flaky on some systems if signal handling is slow.
    original_cwd = os.getcwd()
    try:
        os.chdir(git_repo)
        mock_storage = MockStorageBackend(remote_dir=tmp_path / "remote_storage")
        runs_dir = git_repo / "experiments" / "test-exp" / "runs"

        with pytest.raises(SystemExit) as e:
            with ReproducibilityManager(
                output_dir=str(runs_dir),
                backup_storage_backend=mock_storage
            ) as manager:
                (Path(manager.output_dir) / "results.txt").write_text("partial success")
                os.kill(os.getpid(), sig)
                # The process should exit via the signal handler, so this is a failure.
                pytest.fail("Code continued execution after signal-induced exit.")

        assert e.value.code == 1
        assert len(mock_storage.upload_calls) == 1
        
        remote_file = tmp_path / "remote_storage" / "results.txt"
        assert remote_file.read_text() == "partial success"

    finally:
        os.chdir(original_cwd)


def test_git_helpers_no_submodules_or_superprojects(git_repo: Path):
    """Verify git helper functions return empty lists in a simple repo."""
    original_cwd = os.getcwd()
    try:
        os.chdir(git_repo)
        submodules = get_git_submodules_info(output_dir=None)
        superprojects = get_git_superprojects_info(output_dir=None)
        assert submodules == []
        assert superprojects == []
    finally:
        os.chdir(original_cwd)


# def test_local_backend_idempotency(tmp_path: Path):
#     """Verify that re-uploading to the same destination works without error."""
#     source_dir = tmp_path / "source"
#     source_dir.mkdir()
#     (source_dir / "test.txt").write_text("data")
    
#     storage_root = tmp_path / "storage"
#     backend = LocalFileSystemBackend(root_dir=str(storage_root))
    
#     experiment_id = "idempotent-test"
    
#     # First upload
#     backend.upload(str(source_dir), experiment_id)
#     assert (storage_root / experiment_id / "test.txt").exists()
    
#     # Second upload to the same destination should not raise an error
#     try:
#         (source_dir / "test.txt").write_text("updated data")
#         backend.upload(str(source_dir), experiment_id)
#     except Exception as e:
#         pytest.fail(f"Second upload failed with an exception: {e}")
        
#     # Verify content is updated
#     assert (storage_root / experiment_id / "test.txt").read_text() == "updated data"