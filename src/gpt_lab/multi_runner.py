"""
Multi-runner tool for executing multiple experiments with different parameter configurations.

This module provides the core logic for running batches of experiments, either sequentially
or in parallel. It's designed to work with any experiment script that accepts command-line
arguments.
"""

import os
import subprocess
import logging
import itertools
from typing import Dict, Any, List, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

logger = logging.getLogger(__name__)


def _expand_parameters(parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Expands a parameter dictionary where some values are lists into a list of all combinations.
    
    Args:
        parameters: Dictionary where values can be single values or lists of values.
    
    Returns:
        A list of dictionaries, each representing one parameter combination.
    
    Example:
        Input: {"a": 1, "b": [2, 3], "c": [4, 5]}
        Output: [
            {"a": 1, "b": 2, "c": 4},
            {"a": 1, "b": 2, "c": 5},
            {"a": 1, "b": 3, "c": 4},
            {"a": 1, "b": 3, "c": 5}
        ]
    """
    # Separate static parameters from sweep parameters
    static_params = {}
    sweep_params = {}
    
    for key, value in parameters.items():
        if isinstance(value, list):
            sweep_params[key] = value
        else:
            static_params[key] = [value]  # Wrap in list for consistent handling
    
    # Combine both types
    all_params = {**static_params, **sweep_params}
    
    # Generate all combinations
    keys = list(all_params.keys())
    values = [all_params[k] for k in keys]
    combinations = list(itertools.product(*values))
    
    # Convert back to list of dictionaries
    return [dict(zip(keys, combo)) for combo in combinations]


def _build_command(command_config: Dict[str, Any], params: Dict[str, Any]) -> str:
    """
    Builds a shell command string from a command configuration and parameters.
    
    Args:
        command_config: Dictionary containing 'type', 'script', and type-specific options.
        params: Dictionary of command-line arguments to pass to the script.
    
    Returns:
        A complete shell command string ready to execute.
    """
    cmd_type = command_config.get('type', 'python')
    script = command_config['script']
    
    # Build the parameter string
    param_parts = []
    for key, value in params.items():
        # Handle boolean flags
        if isinstance(value, bool):
            if value:
                param_parts.append(f"--{key}")
        else:
            param_parts.append(f"--{key} {value}")
    
    param_string = " ".join(param_parts)
    
    # Build the full command based on type
    if cmd_type == 'python':
        command = f"python {script} {param_string}"
    elif cmd_type == 'torchrun':
        nproc = command_config.get('nproc_per_node', 1)
        # Add other torchrun-specific flags if they exist
        torchrun_flags = []
        if 'nnodes' in command_config:
            torchrun_flags.append(f"--nnodes={command_config['nnodes']}")
        if 'node_rank' in command_config:
            torchrun_flags.append(f"--node_rank={command_config['node_rank']}")
        if 'master_addr' in command_config:
            torchrun_flags.append(f"--master_addr={command_config['master_addr']}")
        if 'master_port' in command_config:
            torchrun_flags.append(f"--master_port={command_config['master_port']}")
        
        torchrun_flag_str = " ".join(torchrun_flags)
        command = f"torchrun --nproc_per_node={nproc} {torchrun_flag_str} {script} {param_string}".strip()
    elif cmd_type == 'sbatch':
        # For sbatch, we might want to write a temporary script file
        # For now, we'll just build a simple command
        sbatch_flags = command_config.get('sbatch_flags', '')
        command = f"sbatch {sbatch_flags} --wrap='python {script} {param_string}'"
    else:
        raise ValueError(f"Unsupported command type: {cmd_type}")
    
    return command.strip()


def _execute_command(command: str, run_name: str = None) -> Tuple[str, int, str, str]:
    """
    Executes a shell command and returns the result.
    
    Args:
        command: The shell command to execute.
        run_name: Optional name for logging purposes.
    
    Returns:
        A tuple of (run_name, return_code, stdout, stderr).
    """
    name = run_name or command[:50]
    logger.info(f"Starting run: {name}")
    logger.debug(f"Command: {command}")
    
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            logger.info(f"Completed run: {name}")
        else:
            logger.error(f"Failed run: {name} (exit code {result.returncode})")
            logger.error(f"stderr: {result.stderr}")
        
        return name, result.returncode, result.stdout, result.stderr
    
    except Exception as e:
        logger.error(f"Exception during run {name}: {e}")
        return name, -1, "", str(e)


def run_multi(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main function to execute a multi-run configuration.
    
    Args:
        config: A dictionary containing the multi-run configuration.
    
    Returns:
        A summary dictionary with information about the completed runs.
    """
    name = config.get('name', 'unnamed_multi_run')
    logger.info(f"Starting multi-run: {name}")
    
    # Extract configuration sections
    command_config = config['command']
    parameters = config['parameters']
    execution_config = config.get('execution', {'mode': 'sequential'})
    
    # Expand parameter grid
    param_combinations = _expand_parameters(parameters)
    logger.info(f"Generated {len(param_combinations)} parameter combinations")
    
    # Build all commands
    commands = []
    for i, params in enumerate(param_combinations):
        command = _build_command(command_config, params)
        # Create a descriptive name for this run
        run_name = f"{name}_run_{i+1}"
        commands.append((command, run_name))
    
    # Execute commands
    mode = execution_config.get('mode', 'sequential')
    results = []
    
    if mode == 'sequential':
        logger.info("Executing runs sequentially")
        for command, run_name in commands:
            result = _execute_command(command, run_name)
            results.append(result)
    
    elif mode == 'parallel':
        max_workers = execution_config.get('max_workers', 2)
        logger.info(f"Executing runs in parallel (max_workers={max_workers})")
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_execute_command, cmd, name): (cmd, name)
                for cmd, name in commands
            }
            
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
    
    else:
        raise ValueError(f"Unsupported execution mode: {mode}")
    
    # Summarize results
    successful = sum(1 for _, code, _, _ in results if code == 0)
    failed = len(results) - successful
    
    summary = {
        'name': name,
        'total_runs': len(results),
        'successful': successful,
        'failed': failed,
        'results': results
    }
    
    logger.info(f"Multi-run complete: {successful}/{len(results)} successful")
    
    return summary