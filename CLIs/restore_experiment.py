import argparse
import os
import sys

from gpt_lab.reproducibility import (
    restore_experiment_state,
    LocalFileSystemBackend,
    BaseStorageBackend,
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
        experiment_id=args.experiment_id, storage_backend=storage, restore_path=args.restore_path
    )


if __name__ == "__main__":
    main()
