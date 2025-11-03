# CLI: Restore Experiment

**File:** `CLIs/restore_experiment.py`

## Overview

The `restore_experiment.py` script restores the exact code state of a past experiment, allowing for perfect reproducibility. It checks out the specific git commit and applies any uncommitted changes that were captured when the original experiment was run.

This tool is critical for debugging, verification, and building upon previous work with confidence.

## Usage

```bash
python CLIs/restore_experiment.py <experiment_id> [OPTIONS]
```

## Key Arguments

### Positional Arguments

-   **`experiment_id`** (required)
    -   **Description**: The unique identifier for the experiment you want to restore. This is typically the relative path to the experiment's output directory.
    -   **Example**: `experiments/my_exp/runs/2025-10-12_14-30-45_a1b2c3d`

### Optional Arguments

-   **`--backend`**
    -   **Description**: The storage backend where the experiment artifacts are stored.
    -   **Default**: `local`
    -   **Choices**: `local` (support for other backends like S3 is planned).

-   **`--restore_path`**
    -   **Description**: A local directory where the experiment artifacts (logs, checkpoints, etc.) will be downloaded.
    -   **Default**: `restored_experiments`

-   **`--storage_root`**
    -   **Description**: (For `local` backend only) The root directory where your experiment artifacts are saved. This should match the `root_dir` used by the `LocalFileSystemBackend`.
    -   **Default**: `experiment_artifacts`

## Example

Suppose you ran an experiment and its output was saved to `experiments/nano_gpt/runs/2025-11-03_10-00-00_f4a3b2c`. The artifacts were also backed up to the default local storage at `./experiment_artifacts`.

To restore your repository to the state of that experiment:

```bash
python CLIs/restore_experiment.py \
    experiments/nano_gpt/runs/2025-11-03_10-00-00_f4a3b2c \
    --storage_root ./experiment_artifacts \
    --restore_path ./restored_run
```

## Workflow

When you run the script, it performs the following steps:

1.  **Safety Check**: It first verifies that your current git working directory is clean (no uncommitted changes). If it's dirty, the script will exit to prevent you from losing work.
2.  **Download Artifacts**: It connects to the specified storage backend and downloads the `git_info.json` file and the `uncommitted_changes.patch` file (if it exists) into the `--restore_path` directory.
3.  **Checkout Commit**: It reads the `commit_hash` from `git_info.json` and runs `git checkout` to move your repository to that exact commit (placing you in a "detached HEAD" state).
4.  **Apply Patch**: If the original run had uncommitted changes, it applies the downloaded `.patch` file to your working directory.

After the script finishes, your code will be identical to the state it was in when the experiment was launched.
