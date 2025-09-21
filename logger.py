import json
import os
import sys
import subprocess
import datetime
from typing import Optional, Dict, Any

import torch


def get_package_versions() -> Dict[str, Any]:
    """Retrieves versions of installed packages."""
    try:
        result = subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"], text=True
        )
        return {
            line.split("==")[0]: line.split("==")[1]
            for line in result.strip().split("\n")
            if "==" in line
        }
    except Exception as e:
        return {"error": f"Could not run 'pip freeze': {e}"}


class ExperimentLogger:
    """
    A flexible logger for deep learning experiments.

    Handles logging to local files and optional console printing.
    It is designed to be distributed-aware, with each process
    writing to its own log file.
    """

    def __init__(
        self,
        log_dir: str,
        rank: int = 0,
        is_main_process: bool = True,
    ):
        """
        Initializes the logger.

        Args:
            log_dir: The directory to save log files in.
            rank: The global rank of the current process in a distributed setup.
            is_main_process: True if this process is the main one (rank 0).
        """
        self.log_dir = log_dir
        self.rank = rank
        self.is_main_process = is_main_process

        os.makedirs(self.log_dir, exist_ok=True)
        log_file_path = os.path.join(self.log_dir, f"log_rank_{self.rank}.jsonl")
        self.log_file = open(log_file_path, "a")

    def log(
        self,
        data: Dict[str, Any],
        print_to_console: bool = False,
    ):
        """
        Logs a dictionary of data.

        Args:
            data: A dictionary of key-value pairs to log.
            print_to_console: If True, prints to console (main process only).
        """
        log_entry = {
            "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
            **data,
        }
        self.log_file.write(json.dumps(log_entry) + "\n")
        self.log_file.flush()

        if self.is_main_process and print_to_console:
            print(f"[LOG] {json.dumps(data)}")

    def info(self, message: str, print_to_console: bool = True):
        """Logs a simple informational message."""
        self.log(
            {"type": "info", "message": message}, print_to_console=print_to_console
        )

    def log_hyperparams(self, params: Dict[str, Any]):
        """Logs hyperparameters."""
        self.log({"type": "hyperparameters", "data": params}, print_to_console=True)

    def log_system_info(self, git_info: Optional[Dict[str, Any]] = None):
        """
        Logs system information, package versions, and git information.
        
        Args:
            git_info: Git information dictionary from ReproducibilityManager.
                     If None, git information will be omitted from the log.
        """
        info = {
            "type": "system_info",
            "data": {
                "python_version": sys.version,
                "torch_version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "device_count": torch.cuda.device_count()
                if torch.cuda.is_available()
                else 0,
                "devices": [
                    torch.cuda.get_device_name(i)
                    for i in range(torch.cuda.device_count())
                ]
                if torch.cuda.is_available()
                else [],
                "package_versions": get_package_versions(),
            },
        }
        
        # Add git information if provided
        if git_info:
            info["data"]["git_info"] = git_info
        
        self.log(info, print_to_console=True)

    def close(self):
        """Closes the log file."""
        if self.log_file and not self.log_file.closed:
            self.log_file.close()

    def __del__(self):
        self.close()