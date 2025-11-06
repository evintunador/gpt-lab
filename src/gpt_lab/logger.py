import json
import logging
import os
import sys
import subprocess
from typing import Dict, Any, Optional, List

import torch


class JsonFormatter(logging.Formatter):
    """
    A custom log formatter that outputs log records as JSON strings.
    This formatter merges the standard log record attributes with any
    extra data provided in the logging call.
    """

    def format(self, record: logging.LogRecord) -> str:
        # Create a dictionary with standard log record attributes
        log_object = {
            "timestamp": self.formatTime(record, self.datefmt),
            "name": record.name,
            "level": record.levelname,
            "message": record.getMessage(),
        }

        # Add any extra data passed to the logger
        if hasattr(record, "__dict__"):
            extra_data = {
                key: value
                for key, value in record.__dict__.items()
                if key
                not in [
                    "args",
                    "asctime",
                    "created",
                    "exc_info",
                    "exc_text",
                    "filename",
                    "funcName",
                    "levelname",
                    "levelno",
                    "lineno",
                    "module",
                    "msecs",
                    "message",
                    "msg",
                    "name",
                    "pathname",
                    "process",
                    "processName",
                    "relativeCreated",
                    "stack_info",
                    "thread",
                    "threadName",
                ]
            }
            if extra_data:
                log_object.update(extra_data)

        return json.dumps(log_object)


# class Whitelist(logging.Filter):
#     """
#     A logging filter that allows only records whose names start with
#     one of the specified prefixes. This is used to silence logs from
#     third-party libraries and focus on application-specific logging.
#     """
#
#     def __init__(self, prefixes: List[str]):
#         super().__init__()
#         self.prefixes = tuple(prefixes)
#
#     def filter(self, record: logging.LogRecord) -> bool:
#         return record.name.startswith(self.prefixes)


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


def get_system_info(git_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Collects system information, package versions, and optional git information.

    Args:
        git_info: Optional git information dictionary from ReproducibilityManager.

    Returns:
        A dictionary containing system details.
    """
    info = {
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device_count": (
            torch.cuda.device_count() if torch.cuda.is_available() else 0
        ),
        "devices": [
            torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
        ]
        if torch.cuda.is_available()
        else [],
        "package_versions": get_package_versions(),
    }
    if git_info:
        info["git_info"] = git_info
    return info


def setup_experiment_logging(
    log_dir: str, rank: int, is_main_process: bool, level=logging.INFO
):
    """
    Configures the root logger for a reproducible experiment.

    This function sets up a unified logging system that:
    1.  Writes all log records from all ranks to a rank-specific JSONL file.
    2.  If on the main process, writes human-readable logs to a .txt file.
    3.  If on the main process, also prints human-readable logs to the console.

    Args:
        log_dir: The directory to save log files in.
        rank: The global rank of the current process.
        is_main_process: True if this process is the main one (rank 0).
        level: The minimum logging level to capture (e.g., logging.INFO).
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Clear any existing handlers to prevent duplicate logging
    root_logger.handlers.clear()

    # Create a filter to only include logs from our project code and the main script
    # project_filter = Whitelist(["gpt_lab", "experiments", "__main__"])

    # Create the log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)

    # File handler for structured JSON logging (for all ranks)
    log_file_path = os.path.join(log_dir, f"log_rank_{rank}.jsonl")
    file_handler = logging.FileHandler(log_file_path, mode="a")
    file_handler.setFormatter(JsonFormatter())
    # file_handler.addFilter(project_filter)
    root_logger.addHandler(file_handler)

    # Console and text file handlers for human-readable output (main process only)
    if is_main_process:
        console_formatter = logging.Formatter(
            "[%(name)s] %(message)s"
        )

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)

        # Text file handler for the main process
        txt_log_path = os.path.join(log_dir, "log.txt")
        txt_file_handler = logging.FileHandler(txt_log_path, mode="a")
        txt_file_handler.setFormatter(console_formatter)
        root_logger.addHandler(txt_file_handler)