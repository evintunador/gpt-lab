from typing import List, Dict, Any, Optional, Type, Callable, Tuple
import os
import sys
import time
import itertools
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import pandas as pd

from gpt_lab.device import get_available_devices
from gpt_lab.catalog_utils import discover_dunder_objects_in_package
from gpt_lab.optimizers.catalog_utils import OptimizerConfig, OptimizerBenchmarkConfig, create_smart_optimizer
from gpt_lab.catalog_bootstrap import get_artifact_root


# --- Path Constants ---
OPTIMIZERS_ROOT = Path(__file__).parent


def create_complex_synthetic_dataset(
    num_samples: int = 16384,
    input_dim: int = 256,
    num_classes: int = 10,
    device: str = "cpu",
    noise_level: float = 0.3
) -> Tuple[DataLoader, DataLoader]:
    """
    Create a challenging synthetic dataset that's hard to learn but fast to generate.
    
    This creates a multi-class classification problem with:
    - High dimensionality with many irrelevant features
    - Non-linear decision boundaries
    - Significant noise and class imbalance
    - Separate train/val splits for proper convergence measurement
    """
    torch.manual_seed(42)
    
    # Create multiple "prototype" vectors for each class
    num_prototypes = 3  # Reduced from 5
    prototypes = torch.randn(num_classes, num_prototypes, input_dim, device=device)
    
    # Add structured variation to prototypes - only first half are informative
    informative_dims = input_dim // 2
    prototypes[:, :, informative_dims:] *= 0.1  # Make second half mostly noise
    
    X_all = []
    y_all = []
    
    # Create imbalanced class distribution
    class_weights = torch.softmax(torch.randn(num_classes), dim=0)
    samples_per_class = (class_weights * num_samples).long()
    samples_per_class[0] += num_samples - samples_per_class.sum()
    
    for class_idx in range(num_classes):
        class_samples = samples_per_class[class_idx].item()
        
        for i in range(class_samples):
            # Choose prototype randomly
            mode_choice = torch.randint(0, num_prototypes, (1,)).item()
            
            # Mix prototypes with bias toward chosen mode
            weights = torch.rand(num_prototypes, device=device)
            weights[mode_choice] *= 2.0  # Reduced from 3.0
            weights = weights / weights.sum()
            
            # Create sample
            sample = torch.sum(prototypes[class_idx] * weights.unsqueeze(-1), dim=0)
            
            # Add simple non-linearity - much faster than before
            sample = sample + 0.1 * torch.sin(sample)  # Single sine wave only
            
            # Add class-specific bias (replaces complex interactions)
            class_bias = torch.ones(input_dim, device=device) * (class_idx - num_classes/2) * 0.1
            sample = sample + class_bias
            
            # Standard Gaussian noise only
            sample = sample + noise_level * torch.randn_like(sample)
            
            X_all.append(sample)
            y_all.append(torch.tensor(class_idx, device=device))
    
    X = torch.stack(X_all)
    y = torch.stack(y_all)
    
    # Add minimal label noise
    label_noise_prob = 0.02  # Reduced from 0.05
    noise_mask = torch.rand(len(y)) < label_noise_prob
    y[noise_mask] = torch.randint(0, num_classes, (noise_mask.sum(),), device=device)
    
    # Shuffle the data
    perm = torch.randperm(len(X))
    X, y = X[perm], y[perm]
    
    # Split into train/val (80/20)
    split_idx = int(0.8 * len(X))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    train_ds = TensorDataset(X_train, y_train)
    val_ds = TensorDataset(X_val, y_val)
    
    train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=128, shuffle=False)
    
    return train_dl, val_dl


def create_benchmark_model(
    input_dim: int = 256,
    hidden_dims: List[int] = [512, 512, 256],
    num_classes: int = 10,
    device: str = "cpu"
) -> nn.Module:
    """Create a fixed-size MLP for benchmarking"""
    layers = []
    
    # Input layer
    layers.append(nn.Linear(input_dim, hidden_dims[0]))
    layers.append(nn.ReLU())
    layers.append(nn.Dropout(0.1))
    
    # Hidden layers
    for i in range(len(hidden_dims) - 1):
        layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(0.1))
    
    # Output layer
    layers.append(nn.Linear(hidden_dims[-1], num_classes))
    
    model = nn.Sequential(*layers).to(device)
    
    # Initialize weights properly
    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight)
            nn.init.zeros_(module.bias)
    
    return model


def benchmark_optimizer(
    optimizer_class: Type[torch.optim.Optimizer],
    config: OptimizerConfig,
    device: str,
    training_steps: int = 10,
) -> Dict[str, Any]:
    """
    Benchmark a single optimizer configuration.
    
    Returns metrics including timing and memory usage.
    """
    torch.manual_seed(42)  # Ensure reproducible results
    
    # Fixed model configuration
    model_config = {
        'input_dim': 256, 
        'hidden_dims': [512, 512, 256], 
        'num_classes': 10
    }
    
    # Create model and data
    model = create_benchmark_model(device=device, **model_config)
    
    # Compile the model for faster execution
    if device != 'cpu':  # torch.compile works best on GPU
        try:
            model = torch.compile(model, mode='max-autotune')
        except Exception as e:
            print(f"[WARNING] Could not compile model: {e}. Proceeding without compilation.")
    
    train_dl, _ = create_complex_synthetic_dataset(
        device=device, 
        input_dim=model_config['input_dim'],
        num_classes=model_config['num_classes']
    )
    
    # Create optimizer using smart creation
    try:
        optimizer = create_smart_optimizer(model, optimizer_class, config)
        optimizer_info = f"{optimizer_class.__name__}"
        if hasattr(optimizer, 'primary_optimizer'):
            optimizer_info += f" (mixed with {type(optimizer.fallback_optimizer).__name__})"
    except Exception as e:
        raise RuntimeError(f"Failed to create optimizer {optimizer_class.__name__}: {e}")
    
    loss_fn = nn.CrossEntropyLoss()
    device_type = torch.device(device).type
    
    # Memory tracking (CUDA only)
    if device_type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    
    # Create compiled training step function for better performance
    def training_step(model, batch_X, batch_y, optimizer, loss_fn):
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = loss_fn(outputs, batch_y)
        loss.backward()
        optimizer.step()
        return loss
    
    # Compile the training step if not on CPU
    if device != 'cpu':
        try:
            training_step = torch.compile(training_step, mode='max-autotune')
        except Exception as e:
            print(f"[WARNING] Could not compile training step: {e}. Proceeding without compilation.")
    
    # Training loop with timing
    model.train()
    step_times = []
    
    # Extended warmup for compiled functions
    warmup_steps = 10 if device != 'cpu' else 5
    for _ in range(warmup_steps):
        for batch_X, batch_y in train_dl:
            training_step(model, batch_X, batch_y, optimizer, loss_fn)
            break  # Just one batch for warmup
    
    # Synchronize before timing
    if device_type == 'cuda': 
        torch.cuda.synchronize()
    elif device_type == 'mps': 
        torch.mps.synchronize()
    
    # Actual benchmark timing
    total_steps = 0
    
    while total_steps < training_steps:
        for batch_X, batch_y in train_dl:
            if total_steps >= training_steps:
                break
                
            # Time the optimization step
            if device_type == 'cuda':
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                
                start_event.record()
                training_step(model, batch_X, batch_y, optimizer, loss_fn)
                end_event.record()
                
                torch.cuda.synchronize()
                step_time = start_event.elapsed_time(end_event)
            else:
                start_time = time.perf_counter()
                training_step(model, batch_X, batch_y, optimizer, loss_fn)
                if device_type == 'mps': 
                    torch.mps.synchronize()
                end_time = time.perf_counter()
                step_time = (end_time - start_time) * 1000  # Convert to ms
            
            step_times.append(step_time)
            total_steps += 1
    
    # Memory measurement (CUDA only)
    peak_memory_gb = None
    if device_type == 'cuda':
        peak_memory_gb = torch.cuda.max_memory_allocated(device) / 1e9
        torch.cuda.empty_cache()
    
    return {
        # Timing metrics
        'avg_step_time_ms': sum(step_times) / len(step_times),
        
        # Memory metrics
        'peak_memory_gb': peak_memory_gb,
    }


def run_optimizer_benchmarks(configs: List[OptimizerBenchmarkConfig], device: str, output_dir: Optional[Path] = None):
    """Run benchmarks for optimizer configurations following the module pattern"""
    
    if output_dir is None:
        output_dir = get_artifact_root() / "optimizers"
    output_dir.mkdir(parents=True, exist_ok=True)
    device_type = torch.device(device).type
    
    for config in tqdm(configs, desc="All Optimizer Benchmark Configs"):
        all_results = []
        
        param_names = list(config.parameter_space.keys())
        param_values = config.parameter_space.values()
        param_combinations = list(itertools.product(*param_values))

        desc = f"Benchmarking {config.optimizer_name} on {device}"
        for combo in tqdm(param_combinations, desc=desc, leave=False):
            params = dict(zip(param_names, combo))
            
            for competitor_name, competitor in config.competitors.items():
                optimizer_class = competitor['class']
                
                try:
                    # Build optimizer config from parameters
                    optimizer_kwargs = config.optimizer_kwargs_builder(params)
                    
                    optimizer_config = OptimizerConfig(
                        optimizer_kwargs=optimizer_kwargs,
                        param_filter=config.param_filter,
                        fallback_optimizer_class=config.fallback_optimizer_class,
                        fallback_kwargs=config.fallback_kwargs
                    )
                    
                    metrics = benchmark_optimizer(
                        optimizer_class, 
                        optimizer_config, 
                        device,
                    )
                    
                    # Store results in tidy format
                    for metric_name, value in metrics.items():
                        if value is not None:  # Skip None values
                            result_row = params.copy()
                            result_row['class'] = competitor_name
                            result_row['device'] = device
                            result_row['measurement'] = metric_name
                            result_row['value'] = value
                            all_results.append(result_row)
                            
                except Exception as e:
                    param_str = ', '.join(f'{k}={v}' for k, v in params.items())
                    tqdm.write(f"[WARNING] Skipping {competitor_name} for combo ({param_str}) on {device} due to error: {e}")

        if not all_results:
            tqdm.write(f"No results generated for {config.optimizer_name} on {device}. Skipping CSV generation.")
            continue

        df = pd.DataFrame(all_results)
        
        csv_filename = f"{config.optimizer_name}_{device_type}.csv"
        csv_path = output_dir / csv_filename
        df.to_csv(csv_path, index=False)
        tqdm.write(f"Saved benchmark data for {config.optimizer_name} to {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run bulk optimizer benchmarks.",
        add_help=True
    )
    parser.add_argument(
        '-o', '--optimizer',
        type=str,
        default=None,
        help="Run benchmark for a specific optimizer by its optimizer_name."
    )
    args = parser.parse_args()

    all_benchmark_configs, discovery_errors = discover_dunder_objects_in_package(
        dunder='__benchmark_config__', 
        object=OptimizerBenchmarkConfig,
        package_name='gpt_lab.optimizers'
    )

    if discovery_errors:
        print("[WARNING] The following files failed to import during discovery and were skipped:")
        for file, err in discovery_errors.items():
            print(f"  - {file}: {err}")
        print("-" * 20)

    if not all_benchmark_configs and not discovery_errors:
        print("No `__benchmark_config__` found in any optimizer files. Nothing to do.")
        sys.exit(0)

    if args.optimizer:
        benchmark_configs = [c for c in all_benchmark_configs if c.optimizer_name == args.optimizer]
        if not benchmark_configs:
            # Check if the requested optimizer failed to import
            for file, err in discovery_errors.items():
                if f"/{args.optimizer}.py" in file or f"\\{args.optimizer}.py" in file:
                    print(f"Error: Could not load benchmark for '{args.optimizer}' because the file '{file}' failed to import.")
                    print(f"Import Error: {err}")
                    sys.exit(1)

            print(f"Error: No benchmark config found with optimizer_name='{args.optimizer}'.")
            available_optimizers = sorted([c.optimizer_name for c in all_benchmark_configs])
            print(f"Available optimizers are: {available_optimizers}")
            sys.exit(1)
    else:
        benchmark_configs = all_benchmark_configs
    
    available_devices, _ = get_available_devices()
    
    print(f"Found {len(benchmark_configs)} benchmark configuration(s) to run.")
    for device in available_devices:
        if 'cpu' in device:
            continue
        print(f"--- Running benchmarks on {device} ---")
        run_optimizer_benchmarks(benchmark_configs, device)
