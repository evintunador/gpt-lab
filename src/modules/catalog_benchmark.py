from typing import List, Sequence, Union, Callable
import os
import sys
import time
import itertools
import argparse

from tqdm import tqdm
import torch
import pandas as pd

from modules.catalog_utils import BenchmarkConfig, get_total_loss
from utils.device import get_available_devices
from utils.testing import discover_dunder_objects


def measure_performance(
    module: torch.nn.Module, 
    inputs: tuple, 
    device: str,
    num_repeats: int = 10
) -> dict:
    """
    Measures forward/backward time and memory for a given module and input.
    Returns only the metrics that were successfully measured.
    """
    device_type = torch.device(device).type

    # Warmup
    for _ in range(5):
        outputs = module(*inputs)
        loss = get_total_loss(outputs)
        if loss is not None and loss.requires_grad:
            loss.backward()
            module.zero_grad(set_to_none=True)
    
    # Synchronize before measurements
    if device_type == 'cuda': torch.cuda.synchronize()
    elif device_type == 'mps': torch.mps.synchronize()

    # --- Time Measurement ---
    fwd_time_ms = -1.0
    if device_type == 'cuda':
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        for _ in range(num_repeats):
            outputs = module(*inputs)
        end_event.record()
        torch.cuda.synchronize()
        fwd_time_ms = start_event.elapsed_time(end_event) / num_repeats
    else:  # MPS and CPU
        start_time = time.perf_counter()
        for _ in range(num_repeats):
            outputs = module(*inputs)
        if device_type == 'mps': torch.mps.synchronize()
        end_time = time.perf_counter()
        fwd_time_ms = (end_time - start_time) * 1000 / num_repeats

    bwd_time_ms = -1.0
    # To measure backward pass, we need to run forward pass first.
    # We will measure fwd+bwd and subtract fwd.
    # this is done bc of graph issues with modules that use torch.compile on mps; probably not an ideal fix
    outputs = module(*inputs)
    loss = get_total_loss(outputs)

    if loss is not None and loss.requires_grad:
        if device_type == 'cuda':
            start_event.record()
            for _ in range(num_repeats):
                outputs = module(*inputs)
                loss = get_total_loss(outputs)
                loss.backward()
            end_event.record()
            torch.cuda.synchronize()
            total_time_ms = start_event.elapsed_time(end_event) / num_repeats
            bwd_time_ms = total_time_ms - fwd_time_ms
        else:  # MPS and CPU
            start_time = time.perf_counter()
            for _ in range(num_repeats):
                outputs = module(*inputs)
                loss = get_total_loss(outputs)
                loss.backward()
            if device_type == 'mps': torch.mps.synchronize()
            end_time = time.perf_counter()
            total_time_ms = (end_time - start_time) * 1000 / num_repeats
            bwd_time_ms = total_time_ms - fwd_time_ms
            
    module.zero_grad(set_to_none=True)
    if device_type == 'cuda': torch.cuda.empty_cache()

    results = {
        'Forward Time (ms)': fwd_time_ms,
        'Backward Time (ms)': bwd_time_ms if bwd_time_ms > 0 else None,
    }

    # --- Memory Measurement (CUDA only) ---
    if device_type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
        outputs = module(*inputs)
        results['Forward Peak Memory (GB)'] = torch.cuda.max_memory_allocated(device) / 1e9

        torch.cuda.reset_peak_memory_stats(device)
        loss = get_total_loss(outputs)
        if loss is not None:
            loss.backward()
        results['Backward Peak Memory (GB)'] = torch.cuda.max_memory_allocated(device) / 1e9

        module.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
    
    return {k: v for k, v in results.items() if v is not None}


def run_benchmarks(configs: List[BenchmarkConfig], device: str, output_dir: str = "src/modules/bench_results"):
    os.makedirs(output_dir, exist_ok=True)
    device_type = torch.device(device).type
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    for config in tqdm(configs, desc="All Benchmark Configs"):
        all_results = []
        
        param_names = list(config.parameter_space.keys())
        param_values = config.parameter_space.values()
        param_combinations = list(itertools.product(*param_values))

        desc = f"Benchmarking {config.module_name} on {device}"
        for combo in tqdm(param_combinations, desc=desc, leave=False):
            params = dict(zip(param_names, combo))
            
            for competitor_name, competitor in config.competitors.items():
                if competitor.module_class is None: continue
                if competitor.tp_config: continue

                try:
                    init_args = config.init_arg_builder(params)
                    module = competitor.module_class(**init_args).to(device)
                    inputs = config.input_provider(init_args, device)
                    
                    if competitor.run_filter and not competitor.run_filter(inputs):
                        continue

                    perf_metrics = measure_performance(module, inputs, device)
                    
                    # Store results in a tidy format
                    for metric_name, value in perf_metrics.items():
                        result_row = params.copy()
                        result_row['class'] = competitor_name
                        result_row['device'] = device
                        result_row['measurement'] = metric_name
                        result_row['value'] = value
                        all_results.append(result_row)

                except Exception as e:
                    param_str = ', '.join(f'{k}={v.__name__ if isinstance(v, type) else v}' for k, v in params.items())
                    tqdm.write(f"[WARNING] Skipping {competitor_name} for combo ({param_str}) on {device} due to error: {e}")

        if not all_results:
            tqdm.write(f"No results generated for {config.module_name} on {device}. Skipping CSV generation.")
            continue

        df = pd.DataFrame(all_results)
        
        # Sanitize dtype column for CSV
        if 'dtype' in df.columns:
            df['dtype'] = df['dtype'].apply(lambda x: str(x).split('.')[-1])

        csv_filename = f"{config.module_name}_{device_type}.csv"
        csv_path = os.path.join(output_dir, csv_filename)
        df.to_csv(csv_path, index=False)
        tqdm.write(f"Saved benchmark data for {config.module_name} to {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run bulk module benchmarks.",
        add_help=True
    )
    parser.add_argument(
        '-m', '--module',
        type=str,
        default=None,
        help="Run benchmark for a specific module by its module_name."
    )
    args = parser.parse_args()

    all_benchmark_configs, discovery_errors = discover_dunder_objects(
        dunder='__benchmark_config__', 
        object=BenchmarkConfig,
        search_folders="src/modules/catalog/"
    )

    if discovery_errors:
        print("[WARNING] The following files failed to import during discovery and were skipped:")
        for file, err in discovery_errors.items():
            print(f"  - {file}: {err}")
        print("-" * 20)

    if not all_benchmark_configs and not discovery_errors:
        print("No `__benchmark_config__` found in any module files. Nothing to do.")
        sys.exit(0)

    if args.module:
        benchmark_configs = [c for c in all_benchmark_configs if c.module_name == args.module]
        if not benchmark_configs:
            # Check if the requested module failed to import
            for file, err in discovery_errors.items():
                if f"/{args.module}.py" in file or f"\\{args.module}.py" in file:
                    print(f"Error: Could not load benchmark for '{args.module}' because the file '{file}' failed to import.")
                    print(f"Import Error: {err}")
                    sys.exit(1)

            print(f"Error: No benchmark config found with module_name='{args.module}'.")
            available_modules = sorted([c.module_name for c in all_benchmark_configs])
            print(f"Available modules are: {available_modules}")
            sys.exit(1)
    else:
        benchmark_configs = all_benchmark_configs
    
    available_devices, _ = get_available_devices()
    
    print(f"Found {len(benchmark_configs)} benchmark configuration(s) to run.")
    for device in available_devices:
        if 'cpu' in device:
            continue
        print(f"--- Running benchmarks on {device} ---")
        run_benchmarks(benchmark_configs, device)