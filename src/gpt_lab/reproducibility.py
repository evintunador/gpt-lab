import json
import os
import shutil
import subprocess
import datetime
import logging
import uuid
import signal
import sys
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import random

import torch 
import numpy as np


logger = logging.getLogger(__name__)


class BaseStorageBackend(ABC):
    """
    Abstract Base Class for an artifact storage backend.
    This defines the interface that all storage backends must implement.
    """

    @abstractmethod
    def upload(self, local_source_dir: str, experiment_id: str, ignore_patterns: Optional[list[str]] = None):
        """Uploads artifacts from a local directory to a destination."""
        pass

    @abstractmethod
    def download(self, experiment_id: str, local_destination_dir: str):
        """Downloads artifacts from a destination to a local directory."""
        pass


class LocalFileSystemBackend(BaseStorageBackend):
    """A default backend that saves artifacts to another local directory."""

    def __init__(self, root_dir: str = "experiments"):
        self.root_dir = os.path.abspath(root_dir)
        os.makedirs(self.root_dir, exist_ok=True)
        print(f"[Storage] LocalFileSystemBackend initialized at: {self.root_dir}")

    def upload(self, local_source_dir: str, experiment_id: str, ignore_patterns: Optional[list[str]] = None):
        destination = os.path.join(self.root_dir, experiment_id)
        
        if os.path.abspath(local_source_dir) == os.path.abspath(destination):
            print(f"[Storage] Artifacts are already in their final destination: {destination}")
            return
            
        # Use shutil's built-in ignore_patterns utility
        ignore = shutil.ignore_patterns(*ignore_patterns) if ignore_patterns else None
        shutil.copytree(local_source_dir, destination, ignore=ignore, dirs_exist_ok=True)
        print(f"[Storage] Artifacts for '{experiment_id}' saved to {destination}")

    def download(self, experiment_id: str, local_destination_dir: str):
        source = os.path.join(self.root_dir, experiment_id)
        if not os.path.exists(source):
            raise FileNotFoundError(f"No artifacts found for experiment '{experiment_id}' at {source}")
        if os.path.exists(local_destination_dir):
            shutil.rmtree(local_destination_dir)
        shutil.copytree(source, local_destination_dir)
        print(f"[Storage] Artifacts for '{experiment_id}' downloaded to {local_destination_dir}")


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


class FileDaemonHook(BaseDaemonHook):
    """
    A daemon hook that creates a watch file on run start and deletes it on run end.
    An external daemon can monitor the watch directory for these files.
    """

    def __init__(self, watch_dir: str = ".watch_runs"):
        self.watch_dir = os.path.abspath(watch_dir)
        os.makedirs(self.watch_dir, exist_ok=True)
        self.watch_file_path: Optional[str] = None
        logger.info(f"[DaemonHook] FileDaemonHook initialized. Watching directory: {self.watch_dir}")

    def on_run_start(self, run_info: Dict[str, Any]):
        """Creates a unique JSON file with run information."""
        unique_id = uuid.uuid4()
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"{timestamp}_{run_info.get('pid', 'unknown_pid')}_{unique_id}.json"
        self.watch_file_path = os.path.join(self.watch_dir, filename)
        
        with open(self.watch_file_path, 'w') as f:
            json.dump(run_info, f, indent=2)
        
        logger.info(f"Daemon hook: Created watch file at {self.watch_file_path}")

    def on_run_end(self):
        """Deletes the watch file to signal a clean exit."""
        if self.watch_file_path and os.path.exists(self.watch_file_path):
            try:
                os.remove(self.watch_file_path)
                logger.info(f"Daemon hook: Removed watch file {self.watch_file_path}")
            except OSError as e:
                logger.error(f"Daemon hook: Error removing watch file {self.watch_file_path}: {e}")
        self.watch_file_path = None


def get_rng_state():
    return {
        'torch': torch.get_rng_state(),
        'numpy': np.random.get_state(),
        'random': random.getstate(),
    }


def get_git_commit_hash() -> Optional[str]:
    """Retrieves the current git commit hash."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def get_git_remote_url() -> Optional[str]:
    """Retrieves the git remote URL."""
    try:
        return subprocess.check_output(
            ["git", "config", "--get", "remote.origin.url"], text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def get_git_branch() -> Optional[str]:
    """Retrieves the current git branch name."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def is_git_dirty() -> bool:
    """Checks if the git working directory has uncommitted changes."""
    try:
        # Check for modified files
        result = subprocess.run(
            ["git", "diff", "--quiet"], 
            capture_output=True
        )
        if result.returncode != 0:
            return True
            
        # Check for untracked files
        result = subprocess.check_output(
            ["git", "ls-files", "--others", "--exclude-standard"], text=True
        )
        return bool(result.strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def create_git_patch() -> Optional[str]:
    """Creates a patch file containing all uncommitted changes."""
    try:
        # Get all changes (staged and unstaged)
        diff_output = subprocess.check_output(
            ["git", "diff", "HEAD"], text=True
        )
        
        # Get untracked files
        untracked_files = subprocess.check_output(
            ["git", "ls-files", "--others", "--exclude-standard"], text=True
        ).strip().split('\n')
        
        # Add untracked files to the patch
        for file_path in untracked_files:
            if file_path:  # Skip empty strings
                try:
                    with open(file_path, 'r') as f:
                        file_content = f.read()
                    diff_output += f"\ndiff --git a/{file_path} b/{file_path}\n"
                    diff_output += f"new file mode 100644\n"
                    diff_output += f"index 0000000..1234567\n"
                    diff_output += f"--- /dev/null\n"
                    diff_output += f"+++ b/{file_path}\n"
                    diff_output += f"@@ -0,0 +1,{len(file_content.split(chr(10)))} @@\n"
                    for line in file_content.split('\n'):
                        diff_output += f"+{line}\n"
                except (IOError, UnicodeDecodeError):
                    # Skip binary or unreadable files
                    pass
        
        return diff_output if diff_output.strip() else None
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


class ReproducibilityManager:
    """
    Manages the reproducibility of an experiment.
    This is the main user-facing context manager for experiment reproducibility.
    """

    def __init__(
        self,
        output_dir: str,
        storage_backend: Optional[BaseStorageBackend] = None,
        is_main_process: bool = True,
        daemon_hook: Optional[BaseDaemonHook] = None,
    ):
        self.output_root_dir = os.path.abspath(output_dir)
        self.is_main_process = is_main_process
        self.storage_backend = storage_backend
        self.daemon_hook = daemon_hook
        self._is_shutting_down = False
        self.original_sigint_handler = None
        self.original_sigterm_handler = None

        if self.storage_backend is None and self.is_main_process:
            self.storage_backend = LocalFileSystemBackend(root_dir=os.getcwd())

        # These will be set in __enter__
        self.output_dir: Optional[str] = None
        self.git_info: Dict[str, Any] = {}
        self.was_dirty: bool = False

    def _signal_handler(self, signum, frame):
        """Custom signal handler for graceful shutdown."""
        if self._is_shutting_down:
            return  # Avoid re-entrant calls
        self._is_shutting_down = True

        signal_name = "SIGINT" if signum == signal.SIGINT else "SIGTERM"
        logger.info(f"\n--- Interrupted by {signal_name}. Saving artifacts before exiting. ---")
        logger.warning(f"Received {signal_name}, attempting graceful shutdown.")

        # Perform cleanup and upload
        self._cleanup_and_upload(exc_type=signal_name)

        # Exit after saving
        sys.exit(1)

    def __enter__(self):
        """Sets up the experiment environment and captures git state."""
        if self.is_main_process:
            print("\n--- Setting up Reproducible Experiment ---")
            logger.info("Initializing reproducibility manager")
            
            commit_hash = get_git_commit_hash()
            remote_url = get_git_remote_url()
            branch = get_git_branch()
            self.was_dirty = is_git_dirty()
            logger.debug(f"Git commit: {commit_hash}")
            logger.debug(f"Git branch: {branch}")
            logger.debug(f"Git dirty: {self.was_dirty}")
            
            # Create GitHub/GitLab URL if possible
            github_url = None
            if commit_hash and remote_url:
                if "github.com" in remote_url:
                    # Convert SSH URL to HTTPS if needed
                    if remote_url.startswith("git@github.com:"):
                        repo_path = remote_url.replace("git@github.com:", "").replace(".git", "")
                        github_url = f"https://github.com/{repo_path}/commit/{commit_hash}"
                    elif "github.com" in remote_url:
                        repo_path = remote_url.split("github.com/")[-1].replace(".git", "")
                        github_url = f"https://github.com/{repo_path}/commit/{commit_hash}"
            
            # Create unique output directory
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            commit_short = commit_hash[:7] if commit_hash else "unknown"
            dir_name = f"{timestamp}_{commit_short}"
            
            self.output_dir = os.path.join(self.output_root_dir, dir_name)
            os.makedirs(self.output_dir, exist_ok=True)
            
            print(f"Experiment output directory: {self.output_dir}")
            logger.info(f"Experiment output directory: {self.output_dir}", extra={"output_dir": self.output_dir})
            
            self.git_info = {
                "commit_hash": commit_hash,
                "branch": branch,
                "remote_url": remote_url,
                "github_url": github_url,
                "was_dirty": self.was_dirty,
            }
            
            # Save git patch if dirty
            if self.was_dirty:
                patch_content = create_git_patch()
                if patch_content:
                    patch_file = os.path.join(self.output_dir, "uncommitted_changes.patch")
                    self.git_info['patch_file'] = patch_file
                    with open(patch_file, 'w') as f:
                        f.write(patch_content)
                    
            logger.info(f"Git state captured", extra={"git_info": self.git_info})
            
            git_info_file = os.path.join(self.output_dir, "git_info.json")
            with open(git_info_file, 'w') as f:
                json.dump(self.git_info, f, indent=2)
            logger.info(f"Saved git info to: {git_info_file}")
            
            # Call daemon hook on start
            if self.daemon_hook:
                run_info = {
                    "pid": os.getpid(),
                    "output_dir": self.output_dir,
                    "start_time_utc": datetime.datetime.utcnow().isoformat(),
                }
                self.daemon_hook.on_run_start(run_info)

            # Register signal handlers for graceful shutdown
            logger.debug("Registering signal handlers for graceful shutdown.")
            self.original_sigint_handler = signal.signal(signal.SIGINT, self._signal_handler)
            self.original_sigterm_handler = signal.signal(signal.SIGTERM, self._signal_handler)
            
        return self

    def _cleanup_and_upload(self, exc_type=None, exc_val=None):
        """Handles daemon hook cleanup and artifact uploading."""
        if not self.is_main_process:
            return

        # Call daemon hook on end, regardless of outcome
        if self.daemon_hook:
            self.daemon_hook.on_run_end()

        if self.storage_backend and self.output_dir:
            if exc_type is not None:
                if exc_type in ("SIGINT", "SIGTERM"):
                    # The message is printed in the handler
                    logger.error(f"Experiment interrupted by {exc_type}. Partial artifacts saved.")
                else:
                    print(f"\n--- Experiment exited with an error. Attempting to save partial artifacts. ---")
                    logger.error(f"Experiment exited with error: {exc_type.__name__}: {exc_val}")

            logger.info("Finalizing experiment artifacts")

            # The experiment ID is the path relative to the CWD for clean storage paths.
            try:
                experiment_id = os.path.relpath(self.output_dir, os.getcwd())
            except ValueError:
                # Fallback for cases like different drives on Windows
                experiment_id = self.output_dir

            try:
                self.storage_backend.upload(
                    local_source_dir=self.output_dir,
                    experiment_id=experiment_id
                )
                logger.info(f"Artifacts uploaded for experiment: {experiment_id}")
            except Exception as e:
                print(f"[Reproducibility] Warning: Failed to upload artifacts: {e}")
                logger.error(f"Failed to upload artifacts: {e}", exc_info=True)

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Cleans up and optionally uploads artifacts."""
        if self._is_shutting_down:
            return  # Shutdown is already being handled by the signal handler

        # Perform cleanup and upload for normal exit or exception
        self._cleanup_and_upload(exc_type, exc_val)

        # Restore original signal handlers
        if self.is_main_process:
            if self.original_sigint_handler:
                signal.signal(signal.SIGINT, self.original_sigint_handler)
            if self.original_sigterm_handler:
                signal.signal(signal.SIGTERM, self.original_sigterm_handler)

    def get_git_info(self) -> Dict[str, Any]:
        """Returns the git information captured for this experiment."""
        return self.git_info.copy()


def restore_experiment_state(
    experiment_id: str,
    storage_backend: BaseStorageBackend,
    restore_path: str = "restored_experiments",
):
    """
    Restores the code and artifacts of a past experiment.

    This function performs the following steps:
    1. Checks if the current git repository is clean.
    2. Downloads the artifacts for the given experiment_id.
    3. Checks out the specific commit the experiment was run on.
    4. Applies the uncommitted changes from the .patch file.

    Args:
        experiment_id: The unique ID of the experiment to restore
                       (e.g., 'my-exp/2025-09-20_14-30-00_a1b2c3d').
        storage_backend: The storage backend where the artifacts are stored.
        restore_path: A local directory to download the artifacts to.
    """
    # 1. Safety Check: Ensure no local work will be lost.
    if is_git_dirty():
        print("\n\033[91mError: Your git working directory is not clean.\033[0m")
        print("Please commit or stash your changes before restoring an experiment.")
        exit(1)

    print(f"--- Restoring experiment: {experiment_id} ---")
    
    original_branch = get_git_branch() or 'HEAD' # Fallback to HEAD if branch can't be determined

    try:
        # 2. Download the artifacts (logs, patch file, git_info.json, etc.)
        local_artifact_dir = os.path.join(restore_path, experiment_id)
        storage_backend.download(experiment_id, local_artifact_dir)
        print(f"Artifacts downloaded to: {local_artifact_dir}")

        # 3. Read the git_info.json file
        git_info_path = os.path.join(local_artifact_dir, "git_info.json")
        with open(git_info_path, 'r') as f:
            git_info = json.load(f)

        commit_hash = git_info.get("commit_hash")
        if not commit_hash:
            raise ValueError("Could not find commit_hash in git_info.json")

        # 4. Checkout the exact commit
        print(f"Checking out commit: {commit_hash}")
        subprocess.run(["git", "checkout", commit_hash], check=True, capture_output=True, text=True)

        # 5. Apply the patch file if it exists
        patch_file = os.path.join(local_artifact_dir, "uncommitted_changes.patch")
        if os.path.exists(patch_file):
            print("Applying uncommitted changes from patch file...")
            # Use --reject to handle potential conflicts gracefully
            subprocess.run(["git", "apply", "--reject", patch_file], check=True, capture_output=True, text=True)

        print("\n\033[92m✅ Success! Your repository is now in the exact state of the experiment.\033[0m")
        print("Logs and checkpoints are available at:", local_artifact_dir)

    except (subprocess.CalledProcessError, ValueError, FileNotFoundError) as e:
        print(f"\n\033[91m❌ Error during restoration: {e}\033[0m")
        print("Attempting to restore original repository state...")
        try:
            subprocess.run(["git", "checkout", original_branch], check=True, capture_output=True, text=True)
            print(f"\033[92m✅ Successfully restored your original branch ('{original_branch}').\033[0m")
            print("Please check your repository for any lingering changes or .rej files from a failed patch.")
        except subprocess.CalledProcessError as cleanup_error:
            print(f"\n\033[91m❌ Automatic cleanup failed: {cleanup_error.stderr}\033[0m")
            print(f"Your repository may be in a detached HEAD state. To manually restore, please run:")
            print(f"  git reset --hard && git checkout {original_branch}")
        exit(1)
    except Exception as e:
        print(f"\n\033[91m❌ An unexpected error occurred: {e}\033[0m")
        exit(1)
