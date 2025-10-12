# About Marimo Notebooks

GPT-Lab uses [Marimo](https://marimo.io/) for interactive notebooks. Marimo is a reactive Python notebook system that offers several advantages over traditional Jupyter notebooks.

## What is Marimo?

Marimo is a modern Python notebook that:
- **Reactive execution**: Cells automatically re-run when dependencies change
- **Pure Python**: Notebooks are stored as Python scripts (`.py` files)
- **Git-friendly**: Easy to version control, diff, and merge
- **Reproducible**: No hidden state or out-of-order execution issues
- **Interactive UI**: Built-in UI components for creating interactive visualizations

## Installing Marimo

Marimo is included in GPT-Lab's development dependencies:

```bash
# Install with dev dependencies
pip install -e '.[dev]'

# Or install separately
pip install marimo
```

## Running Marimo Notebooks

### Start a Notebook

```bash
# Run a specific notebook
marimo edit notebooks/experiment_comparison.py

# Create a new notebook
marimo edit notebooks/my_analysis.py
```

This opens the notebook in your browser (usually at `http://localhost:2718`).

### Run as Script

Marimo notebooks can also run as standalone Python scripts:

```bash
# Run without the UI
python notebooks/experiment_comparison.py
```

### Convert from Jupyter

If you have existing Jupyter notebooks:

```bash
marimo convert notebook.ipynb > notebook.py
```

### Key Concepts

1. **Cells are Functions**: Each `@app.cell` decorator wraps a function
2. **Returns Define Exports**: Values you return are available to other cells
3. **Parameters are Imports**: Function parameters import from other cells
4. **Reactive Execution**: When a cell changes, dependent cells auto-update

## Resources

- **Official Docs**: [https://docs.marimo.io/](https://docs.marimo.io/)
- **GitHub**: [https://github.com/marimo-team/marimo](https://github.com/marimo-team/marimo)
- **Examples**: [https://marimo.io/gallery](https://marimo.io/gallery)