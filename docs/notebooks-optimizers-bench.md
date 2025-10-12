# Optimizer Benchmark Notebook

The `notebooks/optimizers_bench.py` Marimo notebook provides an interactive interface for exploring optimizer benchmark results.

## Overview

This notebook:
- Loads optimizer benchmark CSV files
- Visualizes convergence speed and efficiency
- Compares performance across optimizers
- Analyzes step time vs. accuracy trade-offs
- Generates comparison reports

## Running the Notebook

```bash
# Start the interactive notebook
marimo edit notebooks/optimizers_bench.py
```

## Features

### 1. CSV Selection

Load benchmark results:
- Lists all CSV files in `artifacts/optimizers/`
- Supports multiple file selection
- Shows benchmark metadata

### 2. Performance Metrics

Key metrics visualized:
- **avg_step_time_ms**: Average time per optimization step
- **loss_reduction_pct**: Loss improvement percentage
- **accuracy_improvement_pct**: Accuracy gain percentage
- **loss_reduction_per_ms**: Efficiency metric (loss reduction per unit time)

### 3. Optimizer Comparison

Compare optimizers on:
- Convergence speed
- Final performance
- Computational efficiency
- Memory usage

### 4. Interactive Exploration

- Filter by optimizer type
- Adjust learning rate ranges
- Compare across datasets
- Export results