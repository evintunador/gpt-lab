# Logger

The `gpt_lab.logger` module provides structured logging for distributed ML experiments. It writes JSONL (JSON Lines) files for each process and optional console output for the main process.

## Overview

The logger component is designed to:
- Write structured, machine-parseable logs (JSONL format) for all processes
- Provide human-readable console output on the main process only
- Filter out noisy third-party library logs
- Support custom fields via the `extra` parameter
- Work seamlessly in distributed training scenarios

## Key Components

### `setup_experiment_logging()`

The main entry point for setting up logging in your experiment:

```python
from gpt_lab.logger import setup_experiment_logging

setup_experiment_logging(
    log_dir="/path/to/logs",
    rank=0,
    is_main_process=True,
    level=logging.INFO
)
```

**Parameters:**
- `log_dir`: Directory where log files will be saved
- `rank`: Global rank of the current process (0 for single-process)
- `is_main_process`: Whether this is the main process (controls console output)
- `level`: Minimum logging level (default: `logging.INFO`)

**Behavior:**
- Creates `log_rank_<rank>.jsonl` in the specified directory
- Clears any existing handlers to prevent duplicate logs
- Adds console handler only for the main process
- Filters logs to only include `gpt_lab`, `experiments`, and `__main__` namespaces

### `JsonFormatter`

A custom formatter that outputs log records as JSON objects:

```python
{
    "timestamp": "2025-10-12 14:30:45",
    "name": "gpt_lab.train",
    "level": "INFO",
    "message": "Training step completed",
    "loss": 0.523,
    "step": 100
}
```

Any extra fields passed via the `extra` parameter are automatically included in the JSON output.

### `Whitelist` Filter

A logging filter that only allows logs from specified prefixes. By default, only logs from:
- `gpt_lab.*`
- `experiments.*`
- `__main__`

This keeps your logs focused on your code and filters out verbose third-party library logs.

### `get_system_info()`

Collects comprehensive system and environment information:

```python
from gpt_lab.logger import get_system_info

info = get_system_info(git_info=None)
# Returns:
# {
#     "python_version": "3.10.12",
#     "torch_version": "2.1.0",
#     "cuda_available": True,
#     "device_count": 2,
#     "devices": ["NVIDIA A100", "NVIDIA A100"],
#     "package_versions": {"numpy": "1.24.3", ...},
#     "git_info": {...}  # if provided
# }
```

This is useful for logging experiment environment at the start of training.

## Usage Examples

### Basic Single-Process Logging

```python
import logging
from gpt_lab.logger import setup_experiment_logging

# Setup logging
setup_experiment_logging(
    log_dir="./runs/experiment_1/logs",
    rank=0,
    is_main_process=True
)

# Use standard Python logging
logger = logging.getLogger(__name__)
logger.info("Starting training")
logger.info("Epoch complete", extra={"epoch": 1, "loss": 0.45, "accuracy": 0.87})
```

### Logging with Rich Metrics

```python
import logging

logger = logging.getLogger("gpt_lab.train")

for step in range(1000):
    loss = train_step()
    
    # Log with structured data
    logger.info(
        "Training step",
        extra={
            "step": step,
            "loss": loss,
            "learning_rate": scheduler.get_last_lr()[0],
            "tokens_per_second": tps
        }
    )
```

## Analyzing JSONL Logs

Since logs are in JSONL format, they're easy to analyze programmatically:

```python
import json
import pandas as pd

# Read logs into a DataFrame
logs = []
with open("logs/log_rank_0.jsonl", "r") as f:
    for line in f:
        logs.append(json.loads(line))

df = pd.DataFrame(logs)

# Filter and analyze
training_logs = df[df["message"] == "Training step"]
print(training_logs[["step", "loss", "learning_rate"]])

# Plot metrics
import matplotlib.pyplot as plt
plt.plot(training_logs["step"], training_logs["loss"])
plt.xlabel("Step")
plt.ylabel("Loss")
plt.show()
```

## Testing

The logger component includes comprehensive tests in `src/gpt_lab/tests/test_logger.py`:

```bash
pytest src/gpt_lab/tests/test_logger.py -v
```

**Test coverage includes:**
- Rank-specific file creation
- JSON formatting with extra fields
- Whitelist filtering behavior
- Console output only on main process

## Contributing

To contribute to the logger component:

1. **Adding Features**: Extend functionality in `src/gpt_lab/logger.py`
   - Add new formatters for different output formats
   - Extend `get_system_info()` to capture more environment details
   - Add new filters for different logging scenarios

2. **Adding Tests**: Add test cases to `src/gpt_lab/tests/test_logger.py`
   - Test new formatters with various input types
   - Test edge cases (empty logs, very large extra data, etc.)
   - Test integration with distributed setups

3. **Guidelines**:
   - Maintain backward compatibility
   - Keep logs machine-parseable (valid JSON)
   - Minimize dependencies
   - Document any new public APIs
