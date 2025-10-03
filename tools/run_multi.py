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

from gpt_lab.configuration import get_config
from gpt_lab.multi_runner import run_multi

# Set up basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger(__name__)


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
    
    # The get_config function will automatically add the --config argument
    # and handle merging YAML config with CLI overrides
    config = get_config(parser)
    
    # Validate required fields
    required_fields = ['command', 'parameters']
    missing_fields = [field for field in required_fields if field not in config]
    if missing_fields:
        logger.error(f"Configuration is missing required fields: {missing_fields}")
        sys.exit(1)
    
    # Execute the multi-run
    try:
        summary = run_multi(config)
        
        # Print summary
        print("\n" + "="*60)
        print(f"Multi-run '{summary['name']}' completed")
        print(f"Total runs: {summary['total_runs']}")
        print(f"Successful: {summary['successful']}")
        print(f"Failed: {summary['failed']}")
        print("="*60)
        
        # Exit with error code if any runs failed
        if summary['failed'] > 0:
            sys.exit(1)
    
    except Exception as e:
        logger.error(f"Multi-run failed with error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()