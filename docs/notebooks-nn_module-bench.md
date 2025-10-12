# nn.Module Benchmark Notebook

The `notebooks/nn_modules_bench.py` Marimo notebook provides an interactive interface for exploring neural network module benchmark results.

## Overview

This notebook:
- Loads benchmark CSV files from the artifacts directory
- Visualizes forward/backward pass timings
- Compares memory usage across modules
- Supports filtering and interactive exploration
- Generates publication-ready plots

## Running the Notebook

```bash
# Start the interactive notebook
marimo edit notebooks/nn_modules_bench.py
```

## Features

### 1. CSV Selection

Select which benchmark files to load:
- Lists all CSV files in `artifacts/nn_modules/`
- Support multiple selections for comparison
- Shows file timestamps and sizes

### 2. Performance Metrics

Visualize key metrics:
- **Forward Time (ms)**: Time to compute forward pass
- **Backward Time (ms)**: Time to compute gradients
- **Forward Peak Memory (GB)**: Maximum memory during forward
- **Backward Peak Memory (GB)**: Maximum memory during backward

### 3. Interactive Filtering

Filter results by:
- Module type
- Device (CPU, CUDA)
- Input shape
- Batch size

### 4. Comparison Plots

- Bar charts comparing modules
- Line plots showing scaling behavior
- Scatter plots for efficiency analysis
- Memory vs. time trade-off plots