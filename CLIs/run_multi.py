"""
Command-line tool for executing multiple experiment runs with different configurations.

This script provides a user-friendly interface to the multi-runner functionality,
allowing researchers to easily launch batches of experiments from the terminal.

Example usage:
    python tools/run_multi.py --config experiments/nano_gpt/multi_run.yaml
    python tools/run_multi.py --config experiments/nano_gpt/multi_run.yaml --execution.mode sequential
"""

import argparse
import logging
import sys

from gpt_lab.configuration import compose_config
from gpt_lab.multi_runner import run_multi

# Set up basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger(__name__)


def validate_config(config: dict) -> list[str]:
    """Validates the configuration has required fields.
    
    Args:
        config: Configuration dictionary to validate
        
    Returns:
        List of missing required field names (empty if valid)
    """
    required_fields = ['command', 'parameters']
    missing_fields = [field for field in required_fields if field not in config]
    return missing_fields


def format_summary(summary: dict) -> str:
    """Formats the multi-run summary into a string.
    
    Args:
        summary: Summary dictionary from run_multi
        
    Returns:
        Formatted summary string
    """
    lines = [
        "=" * 60,
        f"Multi-run '{summary['name']}' completed",
        f"Total runs: {summary['total_runs']}",
        f"Successful: {summary['successful']}",
        f"Failed: {summary['failed']}",
        "=" * 60,
    ]
    return "\n" + "\n".join(lines)


def execute_multi_run(config: dict) -> int:
    """Executes the multi-run with the given configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Exit code (0 for success, 1 for failure)
    """
    # Validate required fields
    missing_fields = validate_config(config)
    if missing_fields:
        logger.error(f"Configuration is missing required fields: {missing_fields}")
        return 1
    
    # Execute the multi-run
    try:
        summary = run_multi(config)
        
        # Print summary
        print(format_summary(summary))
        
        # Return error code if any runs failed
        return 1 if summary['failed'] > 0 else 0
    
    except Exception as e:
        logger.error(f"Multi-run failed with error: {e}", exc_info=True)
        return 1


def main():
    parser = argparse.ArgumentParser(
        description="Run multiple experiments with different parameter configurations.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Examples:
  # Run a batch of experiments defined in a config file
  python tools/run_multi.py --config experiments/nano_gpt/multi_run.yaml
  
  # Override execution mode to run sequentially instead of parallel
  python tools/run_multi.py --config experiments/nano_gpt/multi_run.yaml --execution.mode sequential
  
  # Override max workers for parallel execution
  python tools/run_multi.py --config experiments/nano_gpt/multi_run.yaml --execution.max_workers 4

Note: Any parameter in the YAML configuration can be overridden from the command line
using dot notation (e.g., --execution.mode sequential).
"""
    )
    
    # The compose_config function will automatically add the --config argument
    # and handle merging YAML/JSON config with CLI overrides
    config = compose_config(parser)
    
    exit_code = execute_multi_run(config)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()