"""
Results comparator library for analyzing and comparing experiment runs.

This module provides tools for finding, parsing, and extracting data from
experiment run directories. It reads structured JSONL log files to extract
hyperparameters, metrics, and git information for comparison and analysis.

The library includes a STRATEGY_REGISTRY for extensible metric aggregation:
- last_value: Returns the last value in a time series
- best_value: Returns the min/max based on goal (minimize/maximize)
- time_series: Returns the full list of values

Users can extend the registry by adding custom aggregation strategies.
The registry is used for display purposes only - all library functions
return raw data without aggregation.
"""

import os
import json
import glob
import logging
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class LogSchema:
    """Schema defining the structure and conventions of log files."""
    system_info_message: str = "System Information"
    hyperparams_message: str = "Hyperparameters"
    git_info_field: str = "git_info"
    log_filename: str = "log_rank_0.jsonl"
    preferred_step_keys: List[str] = field(
        default_factory=lambda: ["global_step", "training_step", "step", "iteration", "it", "epoch"]
    )


@dataclass
class MetricResult:
    """Represents the complete data for a single metric from one run."""
    display_name: str
    paths: List[str]
    strategy: str
    goal: Optional[str] = None
    values: List[float] = field(default_factory=list)
    steps: List[Optional[int]] = field(default_factory=list)
    step_series: Dict[str, List[Optional[int]]] = field(default_factory=dict)
    selected_step_key: Optional[str] = None
    selected_steps: List[Optional[int]] = field(default_factory=list)


@dataclass
class RunResult:
    """Represents all extracted information for a single experiment run."""
    run_path: str
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, MetricResult] = field(default_factory=dict)
    git_info: Optional[Dict[str, Any]] = None


def _strategy_last_value(values: List[float]) -> Optional[float]:
    """Returns the last value in the list, or None if empty."""
    return values[-1] if values else None


def _strategy_best_value(values: List[float], goal: str = "minimize") -> Optional[float]:
    """
    Returns the best value based on the goal.
    
    Args:
        values: List of metric values.
        goal: Either "minimize" or "maximize".
    
    Returns:
        The minimum or maximum value, or None if list is empty.
    """
    if not values:
        return None
    if goal == "minimize":
        return min(values)
    elif goal == "maximize":
        return max(values)
    else:
        raise ValueError(f"Unknown goal: {goal}. Expected 'minimize' or 'maximize'.")


def _strategy_time_series(values: List[float]) -> List[float]:
    """Returns the full time series of values."""
    return values


STRATEGY_REGISTRY: Dict[str, Any] = {
    "last_value": _strategy_last_value,
    "best_value": _strategy_best_value,
    "time_series": _strategy_time_series,
}


def _get_nested_value(d: Dict[str, Any], key_path: str) -> Any:
    """
    Retrieves a value from a nested dictionary using a dot-separated path.
    
    Args:
        d: The dictionary to search in.
        key_path: A dot-separated string like "training.learning_rate".
    
    Returns:
        The value at the specified path, or None if not found.
    
    Example:
        >>> d = {"training": {"learning_rate": 0.001}}
        >>> _get_nested_value(d, "training.learning_rate")
        0.001
    """
    keys = key_path.split('.')
    current = d
    
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return None
    
    return current


def _extract_step_value(log_entry: Dict[str, Any], schema: LogSchema) -> Optional[int]:
    """
    Opportunistically extracts a step/iteration counter from a log entry.
    
    Looks for common names like 'step', 'epoch', 'it', 'iteration', etc.
    
    Args:
        log_entry: A parsed JSON log entry.
        schema: LogSchema containing preferred step keys.
    
    Returns:
        The step value as an integer if found, None otherwise.
    """
    for step_name in schema.preferred_step_keys:
        if step_name in log_entry:
            try:
                return int(log_entry[step_name])
            except (ValueError, TypeError):
                continue
    return None


def _extract_all_step_counters(log_entry: Dict[str, Any], schema: LogSchema) -> Dict[str, int]:
    """
    Extracts all present step counters from a log entry.
    
    Args:
        log_entry: A parsed JSON log entry.
        schema: LogSchema containing preferred step keys.
    
    Returns:
        A dictionary mapping counter names to their integer values.
    """
    counters = {}
    for step_name in schema.preferred_step_keys:
        if step_name in log_entry:
            try:
                counters[step_name] = int(log_entry[step_name])
            except (ValueError, TypeError):
                continue
    return counters


def find_run_directories(glob_patterns: List[str], schema: Optional[LogSchema] = None) -> List[str]:
    """
    Finds all valid run directories matching a list of glob patterns.
    
    Args:
        glob_patterns: List of glob patterns (e.g., ["runs/2025-10-04*"]).
        schema: LogSchema defining log file conventions (defaults to LogSchema()).
    
    Returns:
        A sorted list of absolute paths to directories containing the log file.
    """
    if schema is None:
        schema = LogSchema()
    
    matching_dirs = set()
    
    for pattern in glob_patterns:
        # Expand the glob pattern
        for path in glob.glob(pattern):
            # Convert to absolute path
            abs_path = os.path.abspath(path)
            
            # Check if it's a directory and contains the log file
            if os.path.isdir(abs_path):
                log_file = os.path.join(abs_path, schema.log_filename)
                if os.path.exists(log_file):
                    matching_dirs.add(abs_path)
    
    return sorted(list(matching_dirs))


def load_run_data(
    run_dir: str,
    metric_definitions: List[Dict],
    hparam_definitions: List[Dict],
    schema: Optional[LogSchema] = None
) -> RunResult:
    """
    Parses a single run directory and extracts all specified data.
    
    Git information is loaded with the following priority:
    1. git_info.json file in run directory (preferred)
    2. Fallback to git_info field in log entry with System Information message
    3. None if neither source is available
    
    Args:
        run_dir: Path to the experiment run directory.
        metric_definitions: List of dicts defining metrics to extract.
            Example: [{'display_name': 'Validation Loss', 'paths': ['val_loss'],
                      'strategy': 'best_value', 'goal': 'minimize'}]
        hparam_definitions: List of dicts defining hyperparameters to extract.
            Example: [{'display_name': 'Learning Rate',
                      'paths': ['training.learning_rate', 'optim.lr']}]
        schema: LogSchema defining log file conventions (defaults to LogSchema()).
    
    Returns:
        A RunResult object with all extracted data.
    """
    if schema is None:
        schema = LogSchema()
    
    log_file = os.path.join(run_dir, schema.log_filename)
    
    if not os.path.exists(log_file):
        logger.warning(f"Log file not found in {run_dir}")
        return RunResult(run_path=run_dir)
    
    result = RunResult(run_path=run_dir)
    
    # Try to load git_info from git_info.json first
    git_info_json_path = os.path.join(run_dir, 'git_info.json')
    if os.path.exists(git_info_json_path):
        try:
            with open(git_info_json_path, 'r') as f:
                result.git_info = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to load git_info.json from {run_dir}: {e}")
    
    # Temporary storage for metric values during collection
    metric_temp_data = {
        defn['display_name']: {'values': [], 'steps': [], 'step_series': {}, 'definition': defn}
        for defn in metric_definitions
    }
    
    all_hyperparameters = None
    
    # Read and parse the log file
    with open(log_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                log_entry = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping malformed JSON at {log_file}:{line_num}: {e}")
                continue
            
            # Extract git info from system information message (fallback if not loaded from JSON)
            if result.git_info is None and log_entry.get('message') == schema.system_info_message and schema.git_info_field in log_entry:
                result.git_info = log_entry[schema.git_info_field]
            
            # Extract all hyperparameters from hyperparameters message
            if log_entry.get('message') == schema.hyperparams_message:
                all_hyperparameters = {k: v for k, v in log_entry.items()
                                      if k not in ['timestamp', 'name', 'level', 'message', 'taskName']}
            
            # Extract metric values
            step_value = _extract_step_value(log_entry, schema)
            all_step_counters = _extract_all_step_counters(log_entry, schema)
            
            for display_name, temp_data in metric_temp_data.items():
                defn = temp_data['definition']
                # Try each path until we find a match
                for path in defn['paths']:
                    value = _get_nested_value(log_entry, path)
                    if value is not None:
                        try:
                            value = float(value)
                            temp_data['values'].append(value)
                            temp_data['steps'].append(step_value)
                            
                            # Store all step counters for this metric value
                            for counter_name, counter_value in all_step_counters.items():
                                if counter_name not in temp_data['step_series']:
                                    # Initialize with None for all previous values
                                    temp_data['step_series'][counter_name] = [None] * (len(temp_data['values']) - 1)
                                temp_data['step_series'][counter_name].append(counter_value)
                            
                            # For counters not present in this log entry, append None
                            for counter_name in temp_data['step_series']:
                                if counter_name not in all_step_counters:
                                    temp_data['step_series'][counter_name].append(None)
                            
                            break  # Found the metric, don't check other paths
                        except (ValueError, TypeError):
                            continue
    
    # Extract specified hyperparameters
    if all_hyperparameters:
        for hparam_defn in hparam_definitions:
            display_name = hparam_defn['display_name']
            paths = hparam_defn['paths']
            
            # Try each path until we find a value
            for path in paths:
                value = _get_nested_value(all_hyperparameters, path)
                if value is not None:
                    result.hyperparameters[display_name] = value
                    break
    
    # Build MetricResult objects from collected data
    for display_name, temp_data in metric_temp_data.items():
        if temp_data['values']:  # Only include metrics that had data
            defn = temp_data['definition']
            
            # Ensure all step_series have the same length as values
            num_values = len(temp_data['values'])
            for counter_name in temp_data['step_series']:
                while len(temp_data['step_series'][counter_name]) < num_values:
                    temp_data['step_series'][counter_name].append(None)
            
            # Determine selected_step_key using hierarchy
            # First check if metric definition has step_keys override
            step_keys_to_check = defn.get('step_keys', schema.preferred_step_keys)
            
            selected_step_key = None
            for key in step_keys_to_check:
                if key in temp_data['step_series']:
                    selected_step_key = key
                    break
            
            # Get selected_steps based on selected_step_key
            selected_steps = temp_data['step_series'].get(selected_step_key, [None] * num_values) if selected_step_key else [None] * num_values
            
            result.metrics[display_name] = MetricResult(
                display_name=display_name,
                paths=defn['paths'],
                strategy=defn['strategy'],
                goal=defn.get('goal'),
                values=temp_data['values'],
                steps=temp_data['steps'],
                step_series=temp_data['step_series'],
                selected_step_key=selected_step_key,
                selected_steps=selected_steps
            )
    
    return result


def compare_runs(
    run_dirs: List[str],
    metric_definitions: List[Dict],
    hparam_definitions: List[Dict],
    schema: Optional[LogSchema] = None
) -> List[RunResult]:
    """
    Main entry point to parse multiple run directories.
    
    Args:
        run_dirs: List of paths to run directories.
        metric_definitions: List of metric definitions.
        hparam_definitions: List of hyperparameter definitions.
        schema: LogSchema defining log file conventions (defaults to LogSchema()).
    
    Returns:
        A list of RunResult objects, one per directory.
    """
    if schema is None:
        schema = LogSchema()
    
    results = []
    
    for run_dir in run_dirs:
        try:
            run_result = load_run_data(run_dir, metric_definitions, hparam_definitions, schema)
            results.append(run_result)
        except Exception as e:
            logger.error(f"Error loading run from {run_dir}: {e}")
            # Still add a minimal RunResult so the directory is represented
            results.append(RunResult(run_path=run_dir))
    
    return results


if __name__ == "__main__":
    import sys
    import argparse
    import datetime
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Compare experiment runs by extracting metrics and hyperparameters from log files.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--jsonl-logging",
        action="store_true",
        default=False,
        help="Enable JSONL logging of comparator activity to experiments/results_comparator_logs/"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=4,
        help="Maximum number of runs to display (default: 4)"
    )
    args = parser.parse_args()
    
    # Set up human-friendly console logging
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)
    
    # Optionally set up JSONL logging
    if args.jsonl_logging:
        from gpt_lab.logger import setup_experiment_logging
        
        # Create log directory with timestamp
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_dir = os.path.join("experiments", "results_comparator_logs", timestamp)
        os.makedirs(log_dir, exist_ok=True)
        
        # Set up structured JSONL logging
        setup_experiment_logging(log_dir, rank=0, is_main_process=True, level=logging.INFO)
        logger.info("Results comparator started", extra={"log_dir": log_dir})
    
    logger.info("Finding experiment runs...")
    
    # ========================================
    # 1. Create LogSchema (can customize for different log formats)
    # ========================================
    schema = LogSchema(
        system_info_message="System Information",
        hyperparams_message="Hyperparameters",
        git_info_field="git_info",
        log_filename="log_rank_0.jsonl",
        preferred_step_keys=["global_step", "training_step", "step", "iteration", "epoch"]
    )
    logger.info(f"Using schema with log filename: {schema.log_filename}")
    
    # Find runs
    run_patterns = ["experiments/nano_gpt/runs/20*"]
    run_dirs = find_run_directories(run_patterns, schema=schema)
    
    # Limit number of runs for display
    total_found = len(run_dirs)
    run_dirs = run_dirs[:args.limit]
    
    logger.info(f"Found {total_found} runs total, displaying first {len(run_dirs)}")
    
    if not run_dirs:
        print("\nNo runs found. Exiting.")
        sys.exit(0)
    
    print(f"\n{'='*80}")
    print(f"EXPERIMENT RUNS (showing {len(run_dirs)} of {total_found})")
    print(f"{'='*80}")
    for i, run_dir in enumerate(run_dirs, 1):
        print(f"  {i}. {os.path.basename(run_dir)}")
    
    # ========================================
    # 2. Define hyperparameters to extract
    #    These map nested log fields to display names
    # ========================================
    # Raw hyperparameters in logs look like:
    #   {"training": {"learning_rate": 0.001}, "optimizer": {"weight_decay": 0.1}, ...}
    # We extract specific fields with custom display names:
    hparams_to_extract = [
        {"display_name": "Learning Rate", "paths": ["training.learning_rate", "lr"]},  # Try multiple paths
        {"display_name": "Weight Decay", "paths": ["optimizer.weight_decay"]},
        {"display_name": "Batch Size", "paths": ["data.batch_size", "batch_size"]},
    ]
    
    # ========================================
    # 3. Define metrics to extract
    #    Shows nested paths and per-metric step_keys override
    # ========================================
    metrics_to_extract = [
        {
            "display_name": "HellaSwag Accuracy",
            "paths": ["results.accuracy"],  # Nested path: extracts from {"results": {"accuracy": 0.85}}
            "strategy": "best_value",
            "goal": "maximize",
            # Optional: override step_keys for this specific metric
            # "step_keys": ["checkpoint_id", "epoch"]  # Would prefer checkpoint_id over global schema
        },
        {
            "display_name": "WikiQA Accuracy",
            "paths": ["results.accuracy"],
            "strategy": "best_value",
            "goal": "maximize"
        },
        {
            "display_name": "Train Loss",
            "paths": ["train_loss", "loss"],  # Fallback paths
            "strategy": "last_value"
        },
    ]
    
    logger.info(f"Loading data from {len(run_dirs)} runs...")
    
    # Load all run data with schema
    all_run_data = compare_runs(run_dirs, metrics_to_extract, hparams_to_extract, schema=schema)
    
    logger.info("Data loaded successfully")
    
    # ========================================
    # 4. Display results
    # ========================================
    print(f"\n{'='*80}")
    print("COMPARISON RESULTS")
    print(f"{'='*80}")
    print("\nNote: Hyperparameters are extracted from nested log structures")
    print("      (e.g., 'training.learning_rate' from {'training': {'learning_rate': 0.001}})")
    print("      Metrics use strategy registry for aggregation (best, last, series)")
    
    for run in all_run_data:
        print(f"\n{'─'*80}")
        print(f"Run: {os.path.basename(run.run_path)}")
        print(f"{'─'*80}")
        
        # Git information (from git_info.json or logs)
        if run.git_info:
            print(f"  Git:")
            print(f"    Commit: {run.git_info.get('commit_hash', 'N/A')[:7]}")
            print(f"    Branch: {run.git_info.get('branch', 'N/A')}")
            print(f"    Dirty:  {run.git_info.get('was_dirty', 'N/A')}")
        
        # Hyperparameters (mapped from nested paths to display names)
        if run.hyperparameters:
            print(f"  Hyperparameters (extracted via definitions):")
            for name, value in run.hyperparameters.items():
                print(f"    {name:20s} {value}")
        
        # Metrics with step information
        if run.metrics:
            print(f"  Metrics:")
            for name, metric in run.metrics.items():
                if metric.values:
                    # Use strategy registry for aggregation
                    strategy_func = STRATEGY_REGISTRY.get(metric.strategy)
                    
                    if strategy_func is None:
                        print(f"    {name:30s} {len(metric.values)} values (unknown strategy)")
                    elif metric.strategy == 'best_value':
                        agg_value = strategy_func(metric.values, metric.goal or "minimize")
                        if agg_value is not None:
                            goal_str = f"({metric.goal})" if metric.goal else ""
                            # Show step information
                            step_info = f" [step_key: {metric.selected_step_key}]" if metric.selected_step_key else ""
                            print(f"    {name:30s} {agg_value:.4f} {goal_str}{step_info}")
                        else:
                            print(f"    {name:30s} No values")
                    elif metric.strategy == 'last_value':
                        agg_value = strategy_func(metric.values)
                        if agg_value is not None:
                            # Show which step counter is being used
                            step_info = f" [step_key: {metric.selected_step_key}]" if metric.selected_step_key else ""
                            print(f"    {name:30s} {agg_value:.4f} (last){step_info}")
                        else:
                            print(f"    {name:30s} No values")
                    elif metric.strategy == 'time_series':
                        series = strategy_func(metric.values)
                        # Show available step counters
                        step_keys = list(metric.step_series.keys()) if metric.step_series else []
                        step_info = f" [available steps: {', '.join(step_keys)}]" if step_keys else ""
                        print(f"    {name:30s} {len(series)} values in series{step_info}")
                    else:
                        print(f"    {name:30s} {len(metric.values)} values")
    
    print(f"\n{'='*80}")
    logger.info("Comparison complete")
    
    if args.jsonl_logging:
        logger.info("JSONL logs written", extra={"log_dir": log_dir})

