import json
import os
import shutil
import subprocess
import datetime
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any


class BaseStorageBackend(ABC):
    """
    Abstract Base Class for an artifact storage backend.
    This defines the interface that all storage backends must implement.
    """

    @abstractmethod
    def upload(self, local_source_dir: str, experiment_id: str):
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

    def upload(self, local_source_dir: str, experiment_id: str):
        destination = os.path.join(self.root_dir, experiment_id)
        
        if os.path.abspath(local_source_dir) == os.path.abspath(destination):
            print(f"[Storage] Artifacts are already in their final destination: {destination}")
            return
            
        if os.path.exists(destination):
            shutil.rmtree(destination)
        shutil.copytree(local_source_dir, destination)
        print(f"[Storage] Artifacts for '{experiment_id}' saved to {destination}")

    def download(self, experiment_id: str, local_destination_dir: str):
        source = os.path.join(self.root_dir, experiment_id)
        if not os.path.exists(source):
            raise FileNotFoundError(f"No artifacts found for experiment '{experiment_id}' at {source}")
        if os.path.exists(local_destination_dir):
            shutil.rmtree(local_destination_dir)
        shutil.copytree(source, local_destination_dir)
        print(f"[Storage] Artifacts for '{experiment_id}' downloaded to {local_destination_dir}")


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
    ):
        self.output_root_dir = os.path.abspath(output_dir)
        self.is_main_process = is_main_process
        self.storage_backend = storage_backend

        if self.storage_backend is None and self.is_main_process:
            self.storage_backend = LocalFileSystemBackend(root_dir=output_dir)

        # These will be set in __enter__
        self.output_dir: Optional[str] = None
        self.git_info: Dict[str, Any] = {}
        self.was_dirty: bool = False

    def __enter__(self):
        """Sets up the experiment environment and captures git state."""
        if self.is_main_process:
            print("\n--- Setting up Reproducible Experiment ---")
            
            # Capture git information
            commit_hash = get_git_commit_hash()
            remote_url = get_git_remote_url()
            branch = get_git_branch()
            self.was_dirty = is_git_dirty()
            
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
                "was_dirty": self.was_dirty,
            }
            
            # Create unique output directory
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            commit_short = commit_hash[:7] if commit_hash else "unknown"
            dir_name = f"{timestamp}_{commit_short}"
            
            self.output_dir = os.path.join(self.output_root_dir, dir_name)
            os.makedirs(self.output_dir, exist_ok=True)
            
            print(f"[Reproducibility] Experiment output directory: {self.output_dir}")
            print(f"[Reproducibility] Git commit: {commit_hash}")
            print(f"[Reproducibility] Git branch: {branch}")
            print(f"[Reproducibility] Working directory clean: {not self.was_dirty}")
            
            # Save git patch if dirty
            if self.was_dirty:
                patch_content = create_git_patch()
                if patch_content:
                    patch_file = os.path.join(self.output_dir, "uncommitted_changes.patch")
                    with open(patch_file, 'w') as f:
                        f.write(patch_content)
                    print(f"[Reproducibility] Uncommitted changes saved to: uncommitted_changes.patch")
            
            # Save git info to a JSON file
            import json
            git_info_file = os.path.join(self.output_dir, "git_info.json")
            with open(git_info_file, 'w') as f:
                json.dump(self.git_info, f, indent=2)
            
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Cleans up and optionally uploads artifacts."""
        if self.is_main_process and self.storage_backend and self.output_dir:
            if exc_type is not None:
                print(f"\n--- Experiment exited with an error. Attempting to save partial artifacts. ---")
            print("\n--- Uploading Experiment Artifacts ---")
            
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
            except Exception as e:
                print(f"[Reproducibility] Warning: Failed to upload artifacts: {e}")

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

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Restore the state of a previous experiment."
    )
    
    # --- General Arguments ---
    parser.add_argument(
        "experiment_id",
        type=str,
        help="The unique ID of the experiment to restore (e.g., 'my-exp/2025-09-20_14-30-00_a1b2c3d')."
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="local",
        choices=["local"], # Add future backends like 's3' here
        help="The storage backend to use."
    )
    parser.add_argument(
        "--restore_path",
        type=str,
        default="restored_experiments",
        help="The local directory where experiment artifacts will be downloaded."
    )

    # --- Backend-Specific Arguments ---

    # Local File System Backend
    local_group = parser.add_argument_group("Local Backend Arguments")
    local_group.add_argument(
        "--storage_root",
        type=str,
        default="experiment_artifacts",
        help="The root directory of the local file system storage backend."
    )

    # Example for a future S3 Backend
    # s3_group = parser.add_argument_group("S3 Backend Arguments")
    # s3_group.add_argument("--s3-bucket", type=str, help="Name of the S3 bucket.")
    # s3_group.add_argument("--s3-prefix", type=str, default="", help="Optional prefix within the S3 bucket.")


    args = parser.parse_args()

    # --- Initialize Storage Backend ---
    storage: BaseStorageBackend
    if args.backend == "local":
        storage = LocalFileSystemBackend(root_dir=args.storage_root)
    # elif args.backend == 's3':
    #     if not args.s3_bucket:
    #         parser.error("--s3-bucket is required when using the 's3' backend.")
    #     # Assume S3Backend reads credentials from environment variables
    #     storage = S3Backend(bucket_name=args.s3_bucket, prefix=args.s3_prefix)
    else:
        # This will be unreachable until more choices are added
        raise ValueError(f"Unknown backend: {args.backend}")


    restore_experiment_state(
        experiment_id=args.experiment_id,
        storage_backend=storage,
        restore_path=args.restore_path
    )
