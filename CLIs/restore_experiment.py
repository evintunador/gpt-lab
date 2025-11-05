import argparse
import os
import sys

from gpt_lab.reproducibility import (
    restore_experiment_state,
    LocalFileSystemBackend,
    BaseStorageBackend,
)


def create_storage_backend(backend: str, storage_root: str) -> BaseStorageBackend:
    """Creates a storage backend based on the specified type.
    
    Args:
        backend: Type of backend to create ('local', 's3', etc.)
        storage_root: Root directory for the storage backend
        
    Returns:
        Initialized storage backend instance
        
    Raises:
        ValueError: If backend type is unknown
    """
    if backend == "local":
        return LocalFileSystemBackend(root_dir=storage_root)
    # elif backend == 's3':
    #     # Assume S3Backend reads credentials from environment variables
    #     return S3Backend(bucket_name=storage_root, prefix=prefix)
    else:
        raise ValueError(f"Unknown backend: {backend}")


def restore_experiment(
    experiment_id: str,
    backend: str = "local",
    restore_path: str = "restored_experiments",
    storage_root: str = "experiment_artifacts",
):
    """Restores an experiment from a storage backend.
    
    Args:
        experiment_id: The unique ID of the experiment to restore
        backend: Type of storage backend to use
        restore_path: Local directory where artifacts will be downloaded
        storage_root: Root directory of the storage backend
    """
    storage = create_storage_backend(backend, storage_root)
    restore_experiment_state(
        experiment_id=experiment_id,
        storage_backend=storage,
        restore_path=restore_path
    )


def main():
    """Main function to run the CLI tool."""
    parser = argparse.ArgumentParser(description="Restore the state of a previous experiment.")

    # --- General Arguments ---
    parser.add_argument(
        "experiment_id",
        type=str,
        help="The unique ID of the experiment to restore (e.g., 'my-exp/2025-09-20_14-30-00_a1b2c3d').",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="local",
        choices=["local"],  # Add future backends like 's3' here
        help="The storage backend to use.",
    )
    parser.add_argument(
        "--restore_path",
        type=str,
        default="restored_experiments",
        help="The local directory where experiment artifacts will be downloaded.",
    )

    # --- Backend-Specific Arguments ---

    # Local File System Backend
    local_group = parser.add_argument_group("Local Backend Arguments")
    local_group.add_argument(
        "--storage_root",
        type=str,
        default="experiment_artifacts",
        help="The root directory of the local file system storage backend.",
    )

    # Example for a future S3 Backend
    # s3_group = parser.add_argument_group("S3 Backend Arguments")
    # s3_group.add_argument("--s3-bucket", type=str, help="Name of the S3 bucket.")
    # s3_group.add_argument("--s3-prefix", type=str, default="", help="Optional prefix within the S3 bucket.")

    args = parser.parse_args()

    restore_experiment(
        experiment_id=args.experiment_id,
        backend=args.backend,
        restore_path=args.restore_path,
        storage_root=args.storage_root,
    )


if __name__ == "__main__":
    main()
