import json
import os
import subprocess
import datetime
import logging
import signal
import sys
from typing import Optional, Dict, Any
import random

import torch 
import numpy as np

from .daemon_hooks.base import BaseDaemonHook
from .backup_storage_backends.base import BaseBackupStorageBackend
from .distributed import barrier


logger = logging.getLogger(__name__)


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


def get_package_versions() -> Dict[str, Any]:
    """Retrieves versions of installed packages."""
    try:
        result = subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"], text=True
        )
        return {
            line.split("==")[0]: line.split("==")[1]
            for line in result.strip().split("\n")
            if "==" in line
        }
    except Exception as e:
        return {"error": f"Could not run 'pip freeze': {e}"}


def get_system_info(git_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Collects system information, package versions, and optional git information.

    Args:
        git_info: Optional git information dictionary from ReproducibilityManager.

    Returns:
        A dictionary containing system details.
    """
    info = {
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device_count": (
            torch.cuda.device_count() if torch.cuda.is_available() else 0
        ),
        "devices": [
            torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
        ]
        if torch.cuda.is_available()
        else [],
        "package_versions": get_package_versions(),
    }
    if git_info:
        info["git_info"] = git_info
    return info


class ReproducibilityManager:
    """
    Manages the reproducibility of an experiment.
    This is the main user-facing context manager for experiment reproducibility.
    """

    def __init__(
        self,
        output_dir: str,
        backup_storage_backend: Optional[BaseBackupStorageBackend] = None,
        daemon_hook: Optional[BaseDaemonHook] = None,
    ):
        self.output_dir = os.path.abspath(output_dir)
        if self._is_main_process:
            os.makedirs(self.output_dir, exist_ok=True)
            logger.info(f"Experiment output directory set: {self.output_dir}", extra={"output_dir": self.output_dir})

        # Lazily populated caches for additional reproducibility metadata
        self._system_info: Optional[Dict[str, Any]] = None
        self._initial_rng_state: Optional[Dict[str, Any]] = None

        self._get_git_info()
        if self._is_main_process:
            git_info_file = os.path.join(self.output_dir, "git_info.json")
            with open(git_info_file, 'w') as f:
                json.dump(self.git_info, f, indent=2)
            logger.info(f"Saved git info to: {git_info_file}")
            if self.git_info['git_is_dirty'] and self.git_info['patch_content'] is not None:
                with open(self.git_info['patch_file'], 'w') as f:
                    f.write(self.git_info['patch_content'])
                logger.info(f"Saved git patch file to: {self.git_info['patch_file']}")

            # Capture and persist system / environment information
            self._system_info = get_system_info(self.git_info)
            system_info_file = os.path.join(self.output_dir, "system_info.json")
            with open(system_info_file, 'w') as f:
                json.dump(self._system_info, f, indent=2)
            logger.info(f"Saved system info to: {system_info_file}")

            # Capture and persist initial RNG state for reproducibility
            self._initial_rng_state = get_rng_state()
            rng_state_file = os.path.join(self.output_dir, "rng_state_initial.pt")
            torch.save(self._initial_rng_state, rng_state_file)
            logger.info(f"Saved initial RNG state to: {rng_state_file}")
        
        self.pid = os.getpid()
        self.daemon_hook = daemon_hook

        self._is_shutting_down = False
        self.original_sigint_handler = None
        self.original_sigterm_handler = None

        self.backup_storage_backend = backup_storage_backend
        if self.backup_storage_backend is None and self._is_main_process:
            logger.warning(f"No backup storage backend initialized. "
                f"Artifacts in {output_dir} may be lost or corrupted if edited/moved/deleted without a backup.")

    def _get_git_info(self):
        commit_hash = get_git_commit_hash()
        remote_url = get_git_remote_url()
        branch = get_git_branch()
        git_is_dirty = is_git_dirty()
        logger.debug(f"Git commit: {commit_hash}")
        logger.debug(f"Git branch: {branch}")
        logger.debug(f"Git dirty: {git_is_dirty}")

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

        self.git_info = {
            "commit_hash": commit_hash,
            "branch": branch,
            "remote_url": remote_url,
            "github_url": github_url,
            "git_is_dirty": git_is_dirty,
        }

        # Save git patch if dirty
        if git_is_dirty:
            patch_content = create_git_patch()
            self.git_info['patch_content'] = patch_content
            if patch_content:
                patch_file = os.path.join(self.output_dir, "uncommitted_changes.patch")
                self.git_info['patch_file'] = patch_file

        # Log all git info except potentially large "patch_content"
        log_git_info = {k: v for k, v in self.git_info.items() if k != "patch_content"}
        logger.info(f"Git state captured", extra={"git_info": log_git_info})

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
        if self._is_main_process:
            logger.info("Entering reproducibility manager")
            
            if self.daemon_hook:
                self.daemon_hook.on_run_start()                

            logger.debug("Registering signal handlers for graceful shutdown.")
            self.original_sigint_handler = signal.signal(signal.SIGINT, self._signal_handler)
            self.original_sigterm_handler = signal.signal(signal.SIGTERM, self._signal_handler)
            
        return self

    def _cleanup_and_upload(self, exc_type=None, exc_val=None):
        """Handles daemon hook cleanup and artifact uploading."""
        if not self._is_main_process:
            return

        # Call daemon hook on end, regardless of outcome
        if self.daemon_hook:
            self.daemon_hook.on_run_end()

        if self.backup_storage_backend and self.output_dir:
            if exc_type is not None:
                if exc_type in ("SIGINT", "SIGTERM"):
                    # The message is printed in the handler
                    logger.error(f"Experiment interrupted by {exc_type}. Partial artifacts saved.")
                else:
                    print(f"\n--- Experiment exited with an error. Attempting to save partial artifacts. ---")
                    logger.error(f"Experiment exited with error: {exc_type.__name__}: {exc_val}")

            logger.info("Finalizing experiment artifacts")

            try:
                self.backup_storage_backend.upload(source_dir=self.output_dir)
                logger.info(f"Artifacts uploaded to backup storage backend from {self.output_dir}")
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
        if self._is_main_process:
            if self.original_sigint_handler:
                signal.signal(signal.SIGINT, self.original_sigint_handler)
            if self.original_sigterm_handler:
                signal.signal(signal.SIGTERM, self.original_sigterm_handler)

        # Ensure distributed peers wait for main to finish uploads/cleanup
        # No-op in single-process mode
        barrier()

    def get_git_info(self) -> Dict[str, Any]:
        """Returns the git information captured for this experiment."""
        return self.git_info.copy()

    @property
    def system_info(self) -> Dict[str, Any]:
        """Returns captured system / environment information.

        If it was not captured yet on this process (e.g., non-main), it is computed lazily.
        """
        if self._system_info is None:
            # We intentionally do not pass git_info here to avoid implicit file I/O
            self._system_info = get_system_info(self.git_info)
        return self._system_info.copy()

    @property
    def initial_rng_state(self) -> Dict[str, Any]:
        """Returns the RNG state that was captured at experiment start.

        If it was not captured yet on this process (e.g., non-main), returns the current RNG state.
        """
        if self._initial_rng_state is None:
            self._initial_rng_state = get_rng_state()
        return self._initial_rng_state

    def get_rng_states(self) -> Dict[str, Any]:
        """Returns the current RNG states for torch, numpy, and random."""
        return get_rng_state()

    def set_rng_states(self, rng_state: Dict[str, Any]) -> None:
        """Restores RNG states for torch, numpy, and random."""
        torch.set_rng_state(rng_state["torch"])
        np.random.set_state(rng_state["numpy"])
        random.setstate(rng_state["random"])

    @property
    def _is_main_process(self) -> bool:
        """Dynamically reflects whether this process is the main process (rank 0).
        
        Reads directly from the distributed module's shared state so tests that
        tweak `_DIST_STATE` are respected without requiring a real process group.
        """
        # Local import to avoid circulars and to ensure we read the live module state
        from . import distributed as dist_module
        return dist_module._DIST_STATE.get("rank", 0) == 0