import json
import os
import subprocess
import datetime
import logging
import signal
import sys
from typing import Optional, Dict, Any, Union, Callable
import random
import platform
import shutil
import socket
import hashlib

import torch
import numpy as np

from .daemon_hooks.base import BaseDaemonHook
from .backup_storage_backends.base import BaseBackupStorageBackend
from .distributed import barrier


logger = logging.getLogger(__name__)


def get_rng_state():
    """
    Captures the complete RNG state for reproducibility.
    
    Includes:
    - PyTorch CPU RNG state
    - CUDA RNG states for all devices (if CUDA is available)
    - NumPy RNG state
    - Python random module state
    
    This matches what PyTorch Lightning's seed_everything captures.
    """
    state = {
        'torch': torch.get_rng_state(),
        'numpy': np.random.get_state(),
        'random': random.getstate(),
    }
    
    # Capture CUDA RNG states for all devices (if CUDA is available)
    if torch.cuda.is_available():
        state['torch_cuda'] = torch.cuda.get_rng_state_all()
        state['cuda_device_count'] = torch.cuda.device_count()
    else:
        state['torch_cuda'] = None
        state['cuda_device_count'] = 0
    
    return state


def compute_file_hash(filepath: str) -> Optional[str]:
    """Computes the SHA-256 hash of a file."""
    if not os.path.exists(filepath):
        return None
    sha256_hash = hashlib.sha256()
    try:
        with open(filepath, "rb") as f:
            # Read and update hash string value in blocks of 4K
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    except (IOError, OSError):
        return None


def get_git_commit_hash(repo_path: Optional[str] = None) -> Optional[str]:
    """Retrieves the current git commit hash for the given repository."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, cwd=repo_path
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def get_git_remote_url(repo_path: Optional[str] = None) -> Optional[str]:
    """Retrieves the git remote URL for the given repository."""
    try:
        return subprocess.check_output(
            ["git", "config", "--get", "remote.origin.url"], text=True, cwd=repo_path
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def get_git_branch(repo_path: Optional[str] = None) -> Optional[str]:
    """Retrieves the current git branch name for the given repository."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True, cwd=repo_path
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def is_git_dirty(repo_path: Optional[str] = None) -> bool:
    """Checks if the git working directory has uncommitted changes."""
    try:
        # Check for modified files
        result = subprocess.run(
            ["git", "diff", "--quiet"], 
            capture_output=True,
            cwd=repo_path,
        )
        if result.returncode != 0:
            return True
            
        # Check for untracked files
        result = subprocess.check_output(
            ["git", "ls-files", "--others", "--exclude-standard"],
            text=True,
            cwd=repo_path,
        )
        return bool(result.strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def create_git_patch(repo_path: Optional[str] = None) -> Optional[str]:
    """Creates a patch string containing all uncommitted changes for a repo."""
    root_path = repo_path or os.getcwd()
    try:
        # Get all changes (staged and unstaged)
        diff_output = subprocess.check_output(
            ["git", "diff", "HEAD"],
            text=True,
            cwd=root_path,
        )
        
        # Get untracked files
        untracked_files = subprocess.check_output(
            ["git", "ls-files", "--others", "--exclude-standard"],
            text=True,
            cwd=root_path,
        ).strip().split('\n')
        
        # Add untracked files to the patch
        for file_path in untracked_files:
            if file_path:  # Skip empty strings
                try:
                    file_on_disk = os.path.join(root_path, file_path)
                    with open(file_on_disk, 'r') as f:
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


def get_run_invocation_info() -> Dict[str, Any]:
    """
    Captures command-line arguments and a filtered snapshot of environment variables
    relevant to training/distributed execution.
    """
    # Focus on knobs that meaningfully affect runtime behavior; avoid dumping full env.
    prefixes = (
        "CUDA_",
        "NCCL_",
        "OMP_",
        "WORLD_SIZE",
        "RANK",
        "LOCAL_RANK",
        "SLURM_",
        "MASTER_ADDR",
        "MASTER_PORT",
    )
    extra_keys = (
        "PYTHONHASHSEED",
        "PYTHONPATH",
        "VIRTUAL_ENV",
    )
    env_filtered: Dict[str, Any] = {}
    for k, v in os.environ.items():
        if any(k.startswith(prefix) for prefix in prefixes) or k in extra_keys:
            env_filtered[k] = v
    return {
        "argv": list(sys.argv),
        "env": env_filtered,
    }


def get_torch_determinism_info() -> Dict[str, Any]:
    """
    Introspects PyTorch determinism and precision-related knobs.
    """
    info: Dict[str, Any] = {}

    # Deterministic algorithms switch
    try:
        if hasattr(torch, "are_deterministic_algorithms_enabled"):
            info["deterministic_algorithms"] = torch.are_deterministic_algorithms_enabled()
    except Exception as e:  # pragma: no cover - very defensive
        info["deterministic_algorithms_error"] = str(e)

    # Default dtype and float32 matmul precision (PyTorch 2.x+)
    try:
        info["default_dtype"] = str(torch.get_default_dtype())
    except Exception:
        pass

    if hasattr(torch, "get_float32_matmul_precision"):
        try:
            info["float32_matmul_precision"] = torch.get_float32_matmul_precision()
        except Exception:
            pass

    # cuDNN / CUDA backend knobs
    cudnn_info: Dict[str, Any] = {}
    if hasattr(torch.backends, "cudnn"):
        for attr in ("deterministic", "benchmark", "allow_tf32"):
            if hasattr(torch.backends.cudnn, attr):
                cudnn_info[attr] = getattr(torch.backends.cudnn, attr)
    if cudnn_info:
        info["cudnn"] = cudnn_info

    cuda_matmul_info: Dict[str, Any] = {}
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        matmul = torch.backends.cuda.matmul
        for attr in ("allow_tf32", "allow_fp16_reduced_precision_reduction"):
            if hasattr(matmul, attr):
                cuda_matmul_info[attr] = getattr(matmul, attr)
    if cuda_matmul_info:
        info["cuda_matmul"] = cuda_matmul_info

    return info


def get_distributed_topology() -> Dict[str, Any]:
    """
    Returns a lightweight snapshot of the distributed topology.
    """
    is_avail = torch.distributed.is_available()
    is_init = is_avail and torch.distributed.is_initialized()
    
    if is_init:
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
        # Try to infer local_rank from env vars since torch.distributed doesn't expose it directly
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
    else:
        world_size = 1
        rank = 0
        local_rank = 0

    topology: Dict[str, Any] = {
        "is_initialized": is_init,
        "is_distributed": bool(is_init and world_size > 1),
        "world_size": world_size,
        "rank": rank,
        "local_rank": local_rank,
    }

    return topology


def get_software_environment_info() -> Dict[str, Any]:
    """
    Captures the software environment: Python / PyTorch versions, installed packages,
    and PyTorch determinism / precision knobs.
    """
    return {
        "python_version": sys.version,
        "package_versions": get_package_versions(),
        "torch_repro_settings": get_torch_determinism_info(),
    }


def get_runtime_environment_info() -> Dict[str, Any]:
    """
    Captures the runtime environment: devices and distributed topology.
    """
    cuda_available = torch.cuda.is_available()
    device_count = torch.cuda.device_count() if cuda_available else 0

    # Basic device names list (kept for backwards compatibility)
    devices = [
        torch.cuda.get_device_name(i) for i in range(device_count)
    ] if cuda_available else []

    # Richer device properties
    device_properties = []
    if cuda_available:
        for idx in range(device_count):
            try:
                name = torch.cuda.get_device_name(idx)
                capability = torch.cuda.get_device_capability(idx)
                props = torch.cuda.get_device_properties(idx)
                total_memory = getattr(props, "total_memory", None)
            except Exception:
                name = devices[idx] if idx < len(devices) else f"cuda:{idx}"
                capability = None
                total_memory = None

            device_properties.append(
                {
                    "index": idx,
                    "name": name,
                    "capability": capability,
                    "total_memory_bytes": int(total_memory) if total_memory is not None else None,
                }
            )

    # OS / platform information
    os_info = {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }

    # CUDA / driver versions (best-effort)
    cuda_runtime: Dict[str, Any] = {
        "torch_cuda_version": getattr(torch.version, "cuda", None),
        "cudnn_version": None,
        "nvidia_driver_version": None,
    }
    try:
        if hasattr(torch.backends, "cudnn") and hasattr(torch.backends.cudnn, "version"):
            cuda_runtime["cudnn_version"] = torch.backends.cudnn.version()
    except Exception:
        pass

    try:
        # This will fail gracefully on systems without nvidia-smi
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True,
        )
        first_line = out.strip().splitlines()[0].strip() if out.strip() else None
        cuda_runtime["nvidia_driver_version"] = first_line
    except Exception:
        pass

    return {
        "cuda_available": cuda_available,
        "device_count": device_count,
        "devices": devices,
        "device_properties": device_properties,
        "distributed": get_distributed_topology(),
        "os": os_info,
        "cuda_runtime": cuda_runtime,
    }


def _build_github_url(commit_hash: Optional[str], remote_url: Optional[str]) -> Optional[str]:
    """Best-effort construction of a GitHub commit URL from a remote+hash."""
    if not commit_hash or not remote_url:
        return None
    if "github.com" not in remote_url:
        return None

    if remote_url.startswith("git@github.com:"):
        repo_path = remote_url.replace("git@github.com:", "").replace(".git", "")
        return f"https://github.com/{repo_path}/commit/{commit_hash}"
    # HTTPS or other GitHub-style URL
    repo_path = remote_url.split("github.com/")[-1].replace(".git", "")
    return f"https://github.com/{repo_path}/commit/{commit_hash}"


def _sanitize_path_for_patch(path: str) -> str:
    """Sanitize a relative path for safe inclusion in a filename."""
    safe = path.replace(os.sep, "__")
    safe = safe.replace("..", "__")
    return safe or "root"


def get_git_submodules_info(
    output_dir: Optional[str] = None,
    repo_root: Optional[str] = None,
) -> Any:
    """
    Returns git metadata for all submodules (recursively), optionally writing
    patch files for dirty submodules into output_dir.
    """
    root = repo_root or os.getcwd()
    try:
        status = subprocess.check_output(
            ["git", "submodule", "status", "--recursive"],
            text=True,
            cwd=root,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []

    submodules = []

    for line in status.strip().splitlines():
        line = line.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) < 2:
            continue
        # Format: "[+-]SHA path (branch)" – we only need the path here.
        rel_path = parts[1]
        sub_repo_path = os.path.abspath(os.path.join(root, rel_path))

        commit_hash = get_git_commit_hash(sub_repo_path)
        remote_url = get_git_remote_url(sub_repo_path)
        branch = get_git_branch(sub_repo_path)
        git_is_dirty = is_git_dirty(sub_repo_path)
        github_url = _build_github_url(commit_hash, remote_url)

        info: Dict[str, Any] = {
            "path": rel_path,
            "repo_path": sub_repo_path,
            "commit_hash": commit_hash,
            "branch": branch,
            "remote_url": remote_url,
            "github_url": github_url,
            "git_is_dirty": git_is_dirty,
        }

        if git_is_dirty and output_dir is not None:
            patch_content = create_git_patch(sub_repo_path)
            if patch_content:
                safe_name = _sanitize_path_for_patch(rel_path)
                patch_file = os.path.join(
                    output_dir, f"uncommitted_changes.submodule.{safe_name}.patch"
                )
                try:
                    with open(patch_file, "w") as f:
                        f.write(patch_content)
                    info["patch_file"] = patch_file
                    info["patch_file_hash"] = compute_file_hash(patch_file)
                except OSError:
                    # Best-effort; if we can't write the patch file, we just skip it.
                    pass

        submodules.append(info)

    return submodules


def get_git_superprojects_info(output_dir: Optional[str] = None) -> Any:
    """
    Returns git metadata for any enclosing superprojects (if this repo is used
    as a submodule), walking up through nested superprojects if present.
    """
    superprojects = []
    seen_paths = set()
    current_path = os.getcwd()

    while True:
        try:
            super_root = subprocess.check_output(
                ["git", "rev-parse", "--show-superproject-working-tree"],
                text=True,
                cwd=current_path,
            ).strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            break

        if not super_root or super_root in seen_paths:
            break

        seen_paths.add(super_root)

        commit_hash = get_git_commit_hash(super_root)
        remote_url = get_git_remote_url(super_root)
        branch = get_git_branch(super_root)
        git_is_dirty = is_git_dirty(super_root)
        github_url = _build_github_url(commit_hash, remote_url)

        info: Dict[str, Any] = {
            "path": super_root,
            "commit_hash": commit_hash,
            "branch": branch,
            "remote_url": remote_url,
            "github_url": github_url,
            "git_is_dirty": git_is_dirty,
        }

        # Also capture any submodules that live inside this superproject
        submodules = get_git_submodules_info(output_dir=output_dir, repo_root=super_root)
        if submodules:
            info["submodules"] = submodules

        if git_is_dirty and output_dir is not None:
            patch_content = create_git_patch(super_root)
            if patch_content:
                safe_name = _sanitize_path_for_patch(os.path.basename(super_root) or "superproject")
                idx = len(superprojects)
                patch_file = os.path.join(
                    output_dir,
                    f"uncommitted_changes.superproject.{idx}.{safe_name}.patch",
                )
                try:
                    with open(patch_file, "w") as f:
                        f.write(patch_content)
                    info["patch_file"] = patch_file
                    info["patch_file_hash"] = compute_file_hash(patch_file)
                except OSError:
                    pass

        superprojects.append(info)
        current_path = super_root

    return superprojects


class ReproducibilityManager:
    """
    Manages the reproducibility of an experiment.
    This is the main user-facing context manager for experiment reproducibility.
    """

    def __init__(
        self,
        output_dir: str,
        is_main_process: Union[bool, Callable[[], bool]] = None,
        backup_storage_backend: Optional[BaseBackupStorageBackend] = None,
        daemon_hook: Optional[BaseDaemonHook] = None,
    ):
        if is_main_process is None:
            raise ValueError("is_main_process must be explicitly provided (bool or callable).")
            
        if callable(is_main_process):
            self._is_main_process_val = is_main_process()
        else:
            self._is_main_process_val = bool(is_main_process)

        self.output_dir = os.path.abspath(output_dir)

        # Guard against reusing an existing output directory that contains reproducibility artifacts
        if self._is_main_process_val and os.path.exists(self.output_dir):
            # Check for existing "reproducibility" directory
            repro_dir_check = os.path.join(self.output_dir, "reproducibility")
            if os.path.exists(repro_dir_check) and os.listdir(repro_dir_check):
                raise ValueError(
                    f"Output directory '{self.output_dir}' already contains a 'reproducibility' folder. "
                    "To ensure reproducibility, please use a new, unique output directory for each run, "
                    "or ensure the target directory does not contain previous reproducibility artifacts."
                )

        # Unique subdirectory for this process's reproducibility artifacts
        self.hostname = socket.gethostname()
        self.pid = os.getpid()
        self._repro_subdir_name = f"node_{self.hostname}_pid_{self.pid}"
        self._repro_dir = os.path.join(self.output_dir, "reproducibility", self._repro_subdir_name)

        # Ensure directories exist (all processes)
        # Wait for the main process to perform its checks and create the directory structure
        # This prevents a race condition where non-main processes create the directory
        # before the main process checks for its existence/emptiness.
        barrier()
        
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self._repro_dir, exist_ok=True)

        if self._is_main_process_val:
            logger.info(
                f"Experiment output directory set: {self.output_dir}",
                extra={"output_dir": self.output_dir},
            )

        # Lazily populated caches for additional reproducibility metadata
        self._software_environment: Optional[Dict[str, Any]] = None
        self._runtime_environment: Optional[Dict[str, Any]] = None
        self._initial_rng_state: Optional[Dict[str, Any]] = None
        self._run_invocation: Optional[Dict[str, Any]] = None
        self._torch_determinism: Optional[Dict[str, Any]] = None
        self._distributed_topology: Optional[Dict[str, Any]] = None

        self._get_git_info()
        
        # Save artifacts for ALL processes
        # Save canonical git_info.json inside the reproducibility directory
        git_info_file = os.path.join(self._repro_dir, "git_info.json")
        with open(git_info_file, "w") as f:
            json.dump(self.git_info, f, indent=2)
        logger.info(f"Saved git info to: {git_info_file}")

        # Capture and persist software environment information
        self._software_environment = get_software_environment_info()
        software_env_file = os.path.join(self._repro_dir, "software_environment.json")
        with open(software_env_file, "w") as f:
            json.dump(self._software_environment, f, indent=2)
        logger.info(f"Saved software environment info to: {software_env_file}")

        # Capture and persist runtime environment information (devices + distributed)
        self._runtime_environment = get_runtime_environment_info()
        runtime_env_file = os.path.join(self._repro_dir, "runtime_environment.json")
        with open(runtime_env_file, "w") as f:
            json.dump(self._runtime_environment, f, indent=2)
        logger.info(f"Saved runtime environment info to: {runtime_env_file}")

        # Capture and persist initial RNG state for reproducibility
        self._initial_rng_state = get_rng_state()
        rng_state_file = os.path.join(self._repro_dir, "rng_state_initial.pt")
        torch.save(self._initial_rng_state, rng_state_file)
        logger.info(f"Saved initial RNG state to: {rng_state_file}")

        # Capture and persist run-time invocation details (argv + filtered env)
        self._run_invocation = get_run_invocation_info()
        run_invocation_file = os.path.join(self._repro_dir, "run_invocation.json")
        with open(run_invocation_file, "w") as f:
            json.dump(self._run_invocation, f, indent=2)
        logger.info(f"Saved run invocation info to: {run_invocation_file}")
        
        self.daemon_hook = daemon_hook

        self._is_shutting_down = False
        self.original_sigint_handler = None
        self.original_sigterm_handler = None

        self.backup_storage_backend = backup_storage_backend
        if self.backup_storage_backend is None and self._is_main_process_val:
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

        github_url = _build_github_url(commit_hash, remote_url)

        self.git_info = {
            "commit_hash": commit_hash,
            "branch": branch,
            "remote_url": remote_url,
            "github_url": github_url,
            "git_is_dirty": git_is_dirty,
        }

        # Save git patch for the main repository if dirty
        if git_is_dirty and self.output_dir:
            patch_content = create_git_patch()
            if patch_content:
                patch_file = os.path.join(self._repro_dir, "uncommitted_changes.patch")
                try:
                    with open(patch_file, "w") as f:
                        f.write(patch_content)
                    self.git_info["patch_file"] = patch_file
                    self.git_info["patch_file_hash"] = compute_file_hash(patch_file)
                except OSError:
                    logger.warning(f"Failed to write main repo patch file to {patch_file}")

        # Collect submodule and superproject git metadata (and write their patches)
        self.git_info["submodules"] = get_git_submodules_info(self._repro_dir)
        self.git_info["superprojects"] = get_git_superprojects_info(self._repro_dir)

        # Log git info (without any large patch content, which we don't store)
        log_git_info = dict(self.git_info)
        logger.info(f"Git state captured", extra={"git_info": log_git_info})

    def _signal_handler(self, signum, frame):
        """Custom signal handler for graceful shutdown."""
        if self._is_shutting_down:
            return  # Avoid re-entrant calls
        self._is_shutting_down = True

        signal_name = "SIGINT" if signum == signal.SIGINT else "SIGTERM"
        logger.info(f"\n--- Interrupted by {signal_name}. Saving artifacts before exiting. ---")
        logger.warning(f"Received {signal_name}, attempting graceful shutdown.")

        # Best-effort capture of final RNG state before cleanup/upload
        try:
            self._save_final_rng_state()
        except Exception as e:
            logger.error(f"Failed to save final RNG state on signal {signal_name}: {e}", exc_info=True)

        # Perform cleanup and upload
        self._cleanup_and_upload(exc_type=signal_name)

        # Exit after saving
        sys.exit(1)

    def __enter__(self):
        """Sets up the experiment environment and captures git state."""
        if self._is_main_process_val:
            logger.info("Entering reproducibility manager")
            
            # Try to create symlink 'main' -> this process's subdir
            try:
                repro_root = os.path.join(self.output_dir, "reproducibility")
                symlink_path = os.path.join(repro_root, "main")
                # Remove existing if it exists (e.g. from a previous run if output_dir reused)
                if os.path.exists(symlink_path) or os.path.islink(symlink_path):
                    os.remove(symlink_path)
                os.symlink(self._repro_subdir_name, symlink_path)
            except OSError:
                pass # Best effort

            if self.daemon_hook:
                self.daemon_hook.on_run_start()                

            logger.debug("Registering signal handlers for graceful shutdown.")
            self.original_sigint_handler = signal.signal(signal.SIGINT, self._signal_handler)
            self.original_sigterm_handler = signal.signal(signal.SIGTERM, self._signal_handler)
            
        return self

    def _cleanup_and_upload(self, exc_type=None, exc_val=None):
        """Handles daemon hook cleanup and artifact uploading."""
        if not self._is_main_process_val:
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

        # Capture final RNG state at end of context (normal or error exit)
        try:
            self._save_final_rng_state()
        except Exception as e:
            logger.error(f"Failed to save final RNG state on __exit__: {e}", exc_info=True)

        # Perform cleanup and upload for normal exit or exception
        self._cleanup_and_upload(exc_type, exc_val)

        # Restore original signal handlers
        if self._is_main_process_val:
            if self.original_sigint_handler:
                signal.signal(signal.SIGINT, self.original_sigint_handler)
            if self.original_sigterm_handler:
                signal.signal(signal.SIGTERM, self.original_sigterm_handler)

        # Ensure distributed peers wait for main to finish uploads/cleanup
        # No-op in single-process mode
        barrier()

    def _save_final_rng_state(self) -> None:
        """Saves the final RNG state to disk."""
        if not self.output_dir:
            return
        rng_state = get_rng_state()
        rng_state_file = os.path.join(self._repro_dir, "rng_state_final.pt")
        torch.save(rng_state, rng_state_file)
        logger.info(f"Saved final RNG state to: {rng_state_file}")

    def get_git_info(self) -> Dict[str, Any]:
        """Returns the git information captured for this experiment."""
        return self.git_info.copy()

    @property
    def software_environment(self) -> Dict[str, Any]:
        """Returns captured software environment information."""
        if self._software_environment is None:
            self._software_environment = get_software_environment_info()
        return self._software_environment.copy()

    @property
    def runtime_environment(self) -> Dict[str, Any]:
        """Returns captured runtime environment information."""
        if self._runtime_environment is None:
            self._runtime_environment = get_runtime_environment_info()
        return self._runtime_environment.copy()

    @property
    def run_invocation(self) -> Dict[str, Any]:
        """Returns the captured run invocation (argv + filtered env vars)."""
        if self._run_invocation is None:
            self._run_invocation = get_run_invocation_info()
        # Shallow copy is sufficient; nested values are simple types
        return self._run_invocation.copy()

    @property
    def torch_determinism(self) -> Dict[str, Any]:
        """Returns PyTorch determinism/precision settings snapshot."""
        if self._torch_determinism is None:
            self._torch_determinism = get_torch_determinism_info()
        return self._torch_determinism.copy()

    @property
    def distributed_topology(self) -> Dict[str, Any]:
        """Returns the distributed topology snapshot."""
        if self._distributed_topology is None:
            self._distributed_topology = get_distributed_topology()
        return self._distributed_topology.copy()

    @property
    def initial_rng_state(self) -> Dict[str, Any]:
        """Returns the RNG state that was captured at experiment start.

        If it was not captured yet on this process (e.g., non-main), returns the current RNG state.
        """
        if self._initial_rng_state is None:
            self._initial_rng_state = get_rng_state()
        return self._initial_rng_state

    @property
    def final_rng_state(self) -> Optional[Dict[str, Any]]:
        """Returns the RNG state captured at the end of the run, if available."""
        rng_state_file = os.path.join(self._repro_dir, "rng_state_final.pt")
        if os.path.exists(rng_state_file):
            # RNG state includes Python/numpy objects; use weights_only=False.
            return torch.load(rng_state_file, weights_only=False)
        return None

    def get_rng_states(self) -> Dict[str, Any]:
        """Returns the current RNG states for torch, numpy, and random."""
        return get_rng_state()

    def set_rng_states(self, rng_state: Dict[str, Any]) -> None:
        """
        Restores RNG states for torch, numpy, random, and CUDA devices.
        
        This matches PyTorch Lightning's seed_everything behavior by restoring
        all RNG states including CUDA device states.
        """
        torch.set_rng_state(rng_state["torch"])
        np.random.set_state(rng_state["numpy"])
        random.setstate(rng_state["random"])
        
        # Restore CUDA RNG states if available and present in the saved state
        if torch.cuda.is_available() and rng_state.get("torch_cuda") is not None:
            cuda_states = rng_state["torch_cuda"]
            saved_device_count = rng_state.get("cuda_device_count", len(cuda_states))
            current_device_count = torch.cuda.device_count()
            
            # Only restore if the device count matches
            if saved_device_count == current_device_count:
                torch.cuda.set_rng_state_all(cuda_states)
            else:
                logger.warning(
                    f"Cannot restore CUDA RNG states: device count mismatch "
                    f"(saved: {saved_device_count}, current: {current_device_count})"
                )
