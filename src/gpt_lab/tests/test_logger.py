import json
import logging
from pathlib import Path

from gpt_lab.logger import setup_experiment_logging


def test_setup_creates_rank_specific_file(tmp_path: Path):
    """Verify that setup_experiment_logging creates a file with the correct rank."""
    log_dir = tmp_path / "logs"

    # Test main process
    setup_experiment_logging(str(log_dir), rank=0, is_main_process=True)
    assert (log_dir / "log_rank_0.jsonl").exists()

    # Test another process
    setup_experiment_logging(str(log_dir), rank=5, is_main_process=False)
    assert (log_dir / "log_rank_5.jsonl").exists()


def test_file_handler_writes_json_with_extra(tmp_path: Path):
    """Verify that the file logger writes valid JSON with extra data."""
    log_dir = tmp_path / "logs"
    setup_experiment_logging(str(log_dir), rank=0, is_main_process=True)

    logger = logging.getLogger("gpt_lab.test")
    logger.info("Test metric", extra={"loss": 0.123, "step": 1})

    log_file = log_dir / "log_rank_0.jsonl"
    with open(log_file, "r") as f:
        line = f.readline()
    
    data = json.loads(line)
    assert data["name"] == "gpt_lab.test"
    assert data["level"] == "INFO"
    assert data["message"] == "Test metric"
    assert data["loss"] == 0.123
    assert data["step"] == 1


def test_txt_log_created_for_main_process_only(tmp_path: Path):
    """Verify that a human-readable log.txt is created only for the main process."""
    # Test main process
    log_dir_main = tmp_path / "logs_main"
    setup_experiment_logging(str(log_dir_main), rank=0, is_main_process=True)
    assert (log_dir_main / "log.txt").exists()

    # Test worker process
    log_dir_worker = tmp_path / "logs_worker"
    setup_experiment_logging(str(log_dir_worker), rank=1, is_main_process=False)
    assert not (log_dir_worker / "log.txt").exists()


# def test_whitelist_filter_works(tmp_path: Path):
#     """Verify that the Whitelist filter includes and excludes the correct loggers."""
#     log_dir = tmp_path / "logs"
#     setup_experiment_logging(str(log_dir), rank=0, is_main_process=False)

#     # These should be logged
#     logging.getLogger("gpt_lab.utils").info("message 1")
#     logging.getLogger("experiments.run").info("message 2")
#     logging.getLogger("__main__").info("message 3")

#     # This should be filtered out
#     logging.getLogger("third_party.library").warning("message 4")

#     log_file = log_dir / "log_rank_0.jsonl"
#     with open(log_file, "r") as f:
#         lines = f.readlines()
    
#     assert len(lines) == 3
#     assert "message 1" in lines[0]
#     assert "message 2" in lines[1]
#     assert "message 3" in lines[2]


def test_console_handler_main_process_only(tmp_path: Path, capsys):
    """Verify console output only happens on the main process."""
    log_dir = tmp_path / "logs"

    # Case 1: Main process, should print
    setup_experiment_logging(str(log_dir), rank=0, is_main_process=True)
    logging.getLogger("gpt_lab.main").info("Hello from main")
    captured = capsys.readouterr()
    assert "Hello from main" in captured.out

    # Case 2: Worker process, should NOT print
    setup_experiment_logging(str(log_dir), rank=1, is_main_process=False)
    logging.getLogger("gpt_lab.worker").info("Hello from worker")
    captured = capsys.readouterr()
    assert captured.out == ""
