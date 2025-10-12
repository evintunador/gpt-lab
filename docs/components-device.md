# Device

The `gpt_lab.device` module provides utilities for device management, automatically selecting the best available hardware and moving data structures to devices.

## Overview

The device component is designed to:
- Automatically detect and select the best available device (CUDA, MPS, CPU)
- Recursively move complex data structures to devices
- List all available devices for benchmarking
- Support pin memory and non-blocking transfers for optimal performance

## Key Functions

### `get_default_device()`

Returns the best available device, with priority: CUDA > MPS > CPU

```python
from gpt_lab.device import get_default_device
import torch

device = get_default_device()
# Returns: torch.device('cuda:0') if CUDA available
#          torch.device('mps') if Apple Silicon
#          torch.device('cpu') otherwise

model = MyModel().to(device)
```

### `to_device()`

Recursively moves complex data structures to a device:

```python
from gpt_lab.device import to_device
import torch

device = torch.device('cuda:0')

# Works with tensors
tensor = torch.randn(10, 10)
tensor = to_device(tensor, device)

# Works with nested structures
batch = {
    'input_ids': torch.randint(0, 100, (32, 128)),
    'attention_mask': torch.ones(32, 128),
    'labels': torch.randint(0, 10, (32,)),
    'metadata': {
        'embeddings': torch.randn(32, 768),
        'positions': [torch.tensor([0, 1, 2]), torch.tensor([3, 4, 5])]
    }
}
batch = to_device(batch, device)

# Works with modules
model = MyModel()
model = to_device(model, device)

# Supports performance optimizations
tensor = to_device(
    tensor, 
    device='cuda:0',
    pin_memory=True,      # Pin memory for faster CPU->GPU transfers
    non_blocking=True     # Asynchronous transfer when possible
)
```

**Parameters:**
- `item`: Any structure (tensor, module, list, tuple, dict, nested structures)
- `device`: Target device (string or `torch.device`)
- `pin_memory`: Pin memory for CPU tensors moving to CUDA (default: `False`)
- `non_blocking`: Use non-blocking transfer when possible (default: `False`)

**Supported Types:**
- `torch.Tensor`
- `torch.nn.Module`
- `list`, `tuple` (recursively processes elements)
- `dict` (recursively processes values)
- Any object with a `.to()` method

### `get_available_devices()`

Lists all available devices for iteration or benchmarking:

```python
from gpt_lab.device import get_available_devices

# Get all available devices
types, devices = get_available_devices()
# types: ['cuda', 'cpu']
# devices: ['cuda:0', 'cuda:1', 'cpu']

# Exclude specific device types
types, devices = get_available_devices(exclude=['mps'])
# Won't include MPS even if available

# Exclude specific devices
types, devices = get_available_devices(exclude=['cuda:1'])
# Will exclude cuda:1 specifically
```

**Parameters:**
- `exclude`: List of device types or specific devices to exclude

**Returns:**
- Tuple of `(device_types, specific_devices)`
- `device_types`: List of available device type strings (`['cuda', 'cpu']`)
- `specific_devices`: List of specific device strings (`['cuda:0', 'cuda:1', 'cpu']`)

**Device Detection:**
- CUDA: Detects all NVIDIA GPUs via `torch.cuda.device_count()`
- MPS: Detects Apple Silicon
- HIP: Detects AMD ROCm GPUs
- CPU: Always available as fallback

## Testing

Tests for device utilities would typically be in integration tests or benchmarks:

```bash
# Test device utilities as part of full test suite
pytest src/gpt_lab/tests/ -v -k device
```

## Contributing

To contribute to the device component:

1. **Adding Features**: Extend functionality in `src/gpt_lab/device.py`
   - Add support for new hardware backends (TPU, IPU, etc.)
   - Add device memory management utilities
   - Add device-aware autocast helpers
   - Add device topology utilities for multi-GPU

2. **Adding Tests**: Add test cases to appropriate test files
   - Test with different device configurations
   - Test edge cases (empty structures, None values)
   - Test performance optimizations
   - Test cross-device operations

3. **Guidelines**:
   - Keep functions simple and composable
   - Support all PyTorch device types
   - Maintain backward compatibility
   - Handle edge cases gracefully (None, empty containers, etc.)
