# Experiment Comparison Notebook

The `notebooks/experiment_comparison.py` Marimo notebook provides an interactive interface for comparing metrics across multiple experiment runs.

## Overview

This notebook:
- Discovers experiment run directories using glob patterns
- Loads JSONL log files from each run
- Compares metrics across runs
- Visualizes training curves and statistics
- Supports custom metric extraction strategies

## Running the Notebook

```bash
# Start the interactive notebook
marimo edit notebooks/experiment_comparison.py

# Or run as a script (non-interactive)
python notebooks/experiment_comparison.py
```

The notebook opens in your browser at `http://localhost:2718`.

## Features

### 1. Run Discovery

Enter glob patterns to find experiment directories:

```
Input: nano_gpt/runs/*, modded_nano_gpt/runs/*
```

The notebook searches for directories matching these patterns relative to the `experiments/` directory.

### 2. Log Loading

Automatically loads JSONL log files from each run:
- Looks for `logs/log_rank_0.jsonl` in each run directory
- Parses JSON lines into structured data
- Displays run metadata (timestamps, git info)

### 3. Metric Comparison

Compare metrics across runs:
- Select which metrics to compare
- View side-by-side statistics
- Identify best/worst runs for each metric

### 4. Visualization

Interactive plots:
- Training curves over time
- Metric distributions
- Run-to-run comparisons
- Custom aggregations
