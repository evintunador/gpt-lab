import os
import subprocess
import json
from pathlib import Path
import shutil
import pytest
import signal
from unittest.mock import MagicMock, patch
import torch
import socket
import hashlib

from gpt_lab.reproducibility import (
    ReproducibilityManager,
    get_git_submodules_info,
    get_git_superprojects_info,
    compute_file_hash,
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

def test_manager_requires_is_main_process(git_repo: Path):
    """Verify that is_main_process is a required argument."""
    runs_dir = git_repo / "experiments" / "test-exp" / "runs"
    with pytest.raises(ValueError, match="is_main_process must be explicitly provided"):
        ReproducibilityManager(output_dir=str(runs_dir))


def test_manager_clean_repo(git_repo: Path):
    """Verify manager behavior in a clean git repository."""
    original_cwd = os.getcwd()
    try:
        os.chdir(git_repo)
        runs_dir = git_repo / "experiments" / "test-exp" / "runs"
        with ReproducibilityManager(output_dir=str(runs_dir), is_main_process=True) as manager:
            output_dir = Path(manager.output_dir)
            assert output_dir.is_dir()

            # Check for process-specific subdirectory
            hostname = socket.gethostname()
            pid = os.getpid()
            repro_subdir = output_dir / "reproducibility" / f"node_{hostname}_pid_{pid}"
            assert repro_subdir.is_dir()

            # Check git_info.json (now in process-specific subfolder)
            git_info_file = repro_subdir / "git_info.json"
            assert git_info_file.exists()
            with open(git_info_file, "r") as f:
                git_info = json.load(f)
            assert not git_info["git_is_dirty"]
            # New git topology fields should be present, even if empty
            assert "submodules" in git_info
            assert "superprojects" in git_info
            assert isinstance(git_info["submodules"], list)
            assert isinstance(git_info["superprojects"], list)
        
            # Check that no patch file was created
            assert not (repro_subdir / "uncommitted_changes.patch").exists()
            
            # Check for symlink (best effort, might fail on some OS/perms)
            symlink = output_dir / "reproducibility" / "main"
            if symlink.exists():
                assert symlink.is_symlink()
                assert symlink.resolve() == repro_subdir
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
        with ReproducibilityManager(output_dir=str(runs_dir), is_main_process=True) as manager:
            output_dir = Path(manager.output_dir)
            hostname = socket.gethostname()
            pid = os.getpid()
            repro_subdir = output_dir / "reproducibility" / f"node_{hostname}_pid_{pid}"
            
            assert repro_subdir.is_dir()

            # Check git_info.json
            git_info_file = repro_subdir / "git_info.json"
            assert git_info_file.exists()
            with open(git_info_file, "r") as f:
                git_info = json.load(f)
            assert git_info["git_is_dirty"]
        
            # Check that a patch file was created and is not empty
            patch_file = repro_subdir / "uncommitted_changes.patch"
            assert patch_file.exists()
            assert patch_file.read_text().strip() != ""

            # Verify patch file hash is present and correct
            assert "patch_file_hash" in git_info
            assert git_info["patch_file_hash"] is not None
            
            # Verify the hash matches the actual file content
            computed_hash = compute_file_hash(str(patch_file))
            assert git_info["patch_file_hash"] == computed_hash
            assert len(git_info["patch_file_hash"]) == 64 
    finally:
        os.chdir(original_cwd)


def test_manager_non_main_process(git_repo: Path):
    """Verify the manager behavior on non-main processes."""
    original_cwd = os.getcwd()
    try:
        os.chdir(git_repo)
        runs_dir = git_repo / "experiments" / "test-exp" / "runs"
        
        # Non-main process should still create its own repro directory and save artifacts
        with ReproducibilityManager(output_dir=str(runs_dir), is_main_process=False) as manager:
            assert manager._is_main_process_val is False
            
            output_dir = Path(manager.output_dir)
            hostname = socket.gethostname()
            pid = os.getpid()
            repro_subdir = output_dir / "reproducibility" / f"node_{hostname}_pid_{pid}"
            
            assert repro_subdir.is_dir()
            assert (repro_subdir / "git_info.json").exists()
            assert (repro_subdir / "runtime_environment.json").exists()
            assert (repro_subdir / "rng_state_initial.pt").exists()

    finally:
        os.chdir(original_cwd)


def test_manager_callable_is_main(git_repo: Path):
    """Verify that is_main_process can be a callable."""
    runs_dir = git_repo / "experiments" / "test-exp" / "runs"
    
    def is_main():
        return True
        
    with ReproducibilityManager(output_dir=str(runs_dir), is_main_process=is_main) as manager:
        assert manager._is_main_process_val is True


def test_manager_storage_upload(git_repo: Path, tmp_path: Path):
    """Verify that the manager calls the storage backend's upload method."""
    original_cwd = os.getcwd()
    try:
        os.chdir(git_repo)
        mock_storage = MockStorageBackend(remote_dir=tmp_path / "remote_storage")
        
        runs_dir = git_repo / "experiments" / "test-exp" / "runs"
        with ReproducibilityManager(
            output_dir=str(runs_dir),
            is_main_process=True,
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
        with ReproducibilityManager(output_dir=str(runs_dir), is_main_process=True) as manager:
            output_dir = Path(manager.output_dir)
            hostname = socket.gethostname()
            pid = os.getpid()
            repro_subdir = output_dir / "reproducibility" / f"node_{hostname}_pid_{pid}"
            
            assert repro_subdir.is_dir()

            # Software environment
            software_file = repro_subdir / "software_environment.json"
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
            runtime_file = repro_subdir / "runtime_environment.json"
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
        with ReproducibilityManager(output_dir=str(runs_dir), is_main_process=True) as manager:
            output_dir = Path(manager.output_dir)
            hostname = socket.gethostname()
            pid = os.getpid()
            repro_subdir = output_dir / "reproducibility" / f"node_{hostname}_pid_{pid}"
            
            assert repro_subdir.is_dir()

            # File was written
            rng_file = repro_subdir / "rng_state_initial.pt"
            assert rng_file.exists()
            # RNG state includes non-tensor Python/numpy objects; use weights_only=False
            loaded = torch.load(rng_file, weights_only=False)
            assert isinstance(loaded, dict)
            for key in ("torch", "numpy", "random", "torch_cuda", "cuda_device_count"):
                assert key in loaded

            # Properties/methods are available
            initial = manager.initial_rng_state
            assert isinstance(initial, dict)
            current = manager.get_rng_states()
            assert isinstance(current, dict)

            # Verify CUDA RNG states are captured if CUDA is available
            if torch.cuda.is_available():
                assert "torch_cuda" in current
                assert "cuda_device_count" in current
                assert current["cuda_device_count"] == torch.cuda.device_count()
                if current["torch_cuda"] is not None:
                    assert len(current["torch_cuda"]) == current["cuda_device_count"]
            else:
                assert current["torch_cuda"] is None
                assert current["cuda_device_count"] == 0

            # Sanity check set_rng_states works round-trip for torch RNG
            rng_before = manager.get_rng_states()
            _ = torch.rand(1)  # change RNG
            if torch.cuda.is_available():
                _ = torch.cuda.FloatTensor(1).normal_()  # change CUDA RNG
            manager.set_rng_states(rng_before)
            rng_after = manager.get_rng_states()
            assert torch.equal(rng_before["torch"], rng_after["torch"])
            
            # Verify CUDA RNG states are restored if available
            if torch.cuda.is_available() and rng_before["torch_cuda"] is not None:
                assert len(rng_before["torch_cuda"]) == len(rng_after["torch_cuda"])
                for i, (before_state, after_state) in enumerate(zip(rng_before["torch_cuda"], rng_after["torch_cuda"])):
                    assert torch.equal(before_state, after_state), f"CUDA device {i} RNG state mismatch"
            
        # Final RNG state file should exist by the time context exits
        final_rng_file = repro_subdir / "rng_state_final.pt"
        assert final_rng_file.exists()
        final_state = torch.load(final_rng_file, weights_only=False)
        assert isinstance(final_state, dict)
        for key in ("torch", "numpy", "random", "torch_cuda", "cuda_device_count"):
            assert key in final_state

    finally:
        os.chdir(original_cwd)


def test_run_invocation_persisted_and_exposed(git_repo: Path):
    """Verify that run invocation information is saved and exposed."""
    original_cwd = os.getcwd()
    try:
        os.chdir(git_repo)
        runs_dir = git_repo / "experiments" / "test-exp" / "runs"
        with ReproducibilityManager(output_dir=str(runs_dir), is_main_process=True) as manager:
            output_dir = Path(manager.output_dir)
            hostname = socket.gethostname()
            pid = os.getpid()
            repro_subdir = output_dir / "reproducibility" / f"node_{hostname}_pid_{pid}"
            
            assert repro_subdir.is_dir()

            inv_file = repro_subdir / "run_invocation.json"
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


def test_daemon_hook_lifecycle(git_repo: Path):
    """Verify the DaemonHook's start and end methods are called."""
    original_cwd = os.getcwd()
    try:
        os.chdir(git_repo)
        mock_hook = MagicMock()
        
        runs_dir = git_repo / "experiments" / "test-exp" / "runs"
        
        with ReproducibilityManager(
            output_dir=str(runs_dir),
            is_main_process=True,
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
                is_main_process=True,
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
                is_main_process=True,
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


def test_output_dir_guard(git_repo: Path):
    """Verify that reusing an output directory with existing reproducibility artifacts raises an error."""
    runs_dir = git_repo / "experiments" / "test-exp" / "runs"
    
    # First run - creates artifacts
    with ReproducibilityManager(output_dir=str(runs_dir), is_main_process=True):
        pass
        
    # Second run - should fail because reproducibility folder exists and is not empty
    with pytest.raises(ValueError, match="already contains a 'reproducibility' folder"):
        ReproducibilityManager(output_dir=str(runs_dir), is_main_process=True)


def test_output_dir_guard_allows_clean_reuse(git_repo: Path):
    """Verify that reusing an output directory is allowed if it doesn't contain reproducibility artifacts."""
    runs_dir = git_repo / "experiments" / "test-exp" / "runs"
    runs_dir.mkdir(parents=True)
    
    # Create some non-reproducibility files
    (runs_dir / "some_log.txt").write_text("log content")
    
    # Should succeed
    with ReproducibilityManager(output_dir=str(runs_dir), is_main_process=True):
        pass


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


def test_cuda_rng_state_capture_and_restore(git_repo: Path):
    """Verify that CUDA RNG states are properly captured and restored."""
    original_cwd = os.getcwd()
    try:
        os.chdir(git_repo)
        runs_dir = git_repo / "experiments" / "test-exp" / "runs"
        with ReproducibilityManager(output_dir=str(runs_dir), is_main_process=True) as manager:
            rng_state = manager.get_rng_states()
            
            # Verify CUDA fields are always present
            assert "torch_cuda" in rng_state
            assert "cuda_device_count" in rng_state
            
            if torch.cuda.is_available():
                # If CUDA is available, verify states are captured
                assert rng_state["torch_cuda"] is not None
                assert rng_state["cuda_device_count"] == torch.cuda.device_count()
                assert len(rng_state["torch_cuda"]) == torch.cuda.device_count()
                
                # Test round-trip restore for CUDA
                # Generate some random numbers to change CUDA RNG state
                original_states = [torch.cuda.get_rng_state(i) for i in range(torch.cuda.device_count())]
                _ = torch.cuda.FloatTensor(10).normal_()  # Change CUDA RNG
                
                # Restore and verify
                manager.set_rng_states(rng_state)
                restored_states = [torch.cuda.get_rng_state(i) for i in range(torch.cuda.device_count())]
                
                for i, (orig, restored) in enumerate(zip(original_states, restored_states)):
                    assert torch.equal(orig, restored), f"CUDA device {i} RNG state not properly restored"
            else:
                # If CUDA is not available, verify None/0 values
                assert rng_state["torch_cuda"] is None
                assert rng_state["cuda_device_count"] == 0
                
    finally:
        os.chdir(original_cwd)
