import os
import subprocess
import json
from pathlib import Path
import shutil
import pytest
import signal
import sys
from unittest.mock import MagicMock

from gpt_lab.reproducibility import ReproducibilityManager
from gpt_lab.reproducibility.storage_backends.base import BaseStorageBackend


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


class MockStorageBackend(BaseStorageBackend):
    """A mock storage backend that records calls for testing purposes."""
    def __init__(self, source_artifacts_dir: Path):
        self.upload_calls = []
        self.download_calls = []
        self.source_artifacts_dir = source_artifacts_dir

    def upload(self, local_source_dir: str, experiment_id: str):
        self.upload_calls.append((local_source_dir, experiment_id))
        # Simulate the upload by copying to a "remote" location
        remote_path = self.source_artifacts_dir / experiment_id
        if remote_path.exists():
            shutil.rmtree(remote_path)
        shutil.copytree(local_source_dir, remote_path)

    def download(self, experiment_id: str, local_destination_dir: str):
        self.download_calls.append((experiment_id, local_destination_dir))
        # Simulate download by copying from the "remote" location
        remote_path = self.source_artifacts_dir / experiment_id
        if not remote_path.exists():
            raise FileNotFoundError(f"'{experiment_id}' not found in mock remote storage.")
        shutil.copytree(remote_path, local_destination_dir)


# --- Core Functionality Tests ---

def test_manager_clean_repo(git_repo: Path):
    """Verify manager behavior in a clean git repository."""
    original_cwd = os.getcwd()
    try:
        os.chdir(git_repo)
        runs_dir = git_repo / "experiments" / "test-exp" / "runs"
        with ReproducibilityManager(output_dir=str(runs_dir), is_main_process=True) as manager:
            output_dir = Path(manager.output_dir)
            assert output_dir.is_dir()

            # Check git_info.json
            git_info_file = output_dir / "git_info.json"
            assert git_info_file.exists()
            with open(git_info_file, 'r') as f:
                git_info = json.load(f)
            assert not git_info["git_is_dirty"]
        
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
        with ReproducibilityManager(output_dir=str(runs_dir), is_main_process=True) as manager:
            output_dir = Path(manager.output_dir)
            assert output_dir.is_dir()

            # Check git_info.json
            git_info_file = output_dir / "git_info.json"
            assert git_info_file.exists()
            with open(git_info_file, 'r') as f:
                git_info = json.load(f)
            assert git_info["git_is_dirty"]
        
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
        with ReproducibilityManager(
            output_dir=str(runs_dir),
            is_main_process=False
        ) as manager:
            assert manager.is_main_process is False
    finally:
        os.chdir(original_cwd)


def test_manager_storage_upload(git_repo: Path, tmp_path: Path):
    """Verify that the manager calls the storage backend's upload method."""
    original_cwd = os.getcwd()
    try:
        os.chdir(git_repo)
        mock_storage = MockStorageBackend(source_artifacts_dir=tmp_path / "remote_storage")
        
        runs_dir = git_repo / "experiments" / "test-exp" / "runs"
        with ReproducibilityManager(
            output_dir=str(runs_dir),
            is_main_process=True,
            storage_backend=mock_storage
        ) as manager:
            # Simulate creating an output file in the experiment dir
            (Path(manager.output_dir) / "results.txt").write_text("success")

        assert len(mock_storage.upload_calls) == 1
        
        # Check that the uploaded content is correct
        uploaded_dir, experiment_id = mock_storage.upload_calls[0]
        assert "test-exp" in experiment_id
        assert (tmp_path / "remote_storage" / experiment_id / "results.txt").read_text() == "success"
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
            
            # Extract the run_info dict from the call
            call_args, _ = mock_hook.on_run_start.call_args
            run_info = call_args[0]
            
            assert "pid" in run_info
            assert "output_dir" in run_info

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
        mock_storage = MockStorageBackend(source_artifacts_dir=tmp_path / "remote_storage")
        runs_dir = git_repo / "experiments" / "test-exp" / "runs"

        with pytest.raises(SystemExit) as e:
            with ReproducibilityManager(
                output_dir=str(runs_dir),
                is_main_process=True,
                storage_backend=mock_storage
            ) as manager:
                (Path(manager.output_dir) / "results.txt").write_text("partial success")
                os.kill(os.getpid(), sig)
                # The process should exit via the signal handler, so this is a failure.
                pytest.fail("Code continued execution after signal-induced exit.")

        assert e.value.code == 1
        assert len(mock_storage.upload_calls) == 1
        
        uploaded_dir, experiment_id = mock_storage.upload_calls[0]
        assert "test-exp" in experiment_id
        remote_file = tmp_path / "remote_storage" / experiment_id / "results.txt"
        assert remote_file.read_text() == "partial success"
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