import json
from pathlib import Path

from src.utils.logging import ExperimentLogger


def test_logger_creates_file(tmp_path: Path):
    """Verify that the logger creates a file with the correct rank in the name."""
    log_dir = tmp_path / "logs"

    # Test main process
    logger0 = ExperimentLogger(log_dir=str(log_dir), rank=0)
    logger0.close()
    assert (log_dir / "log_rank_0.jsonl").exists()

    # Test another process
    logger5 = ExperimentLogger(log_dir=str(log_dir), rank=5)
    logger5.close()
    assert (log_dir / "log_rank_5.jsonl").exists()


def test_logger_writes_valid_json(tmp_path: Path):
    """Verify that the logger writes valid, single-line JSON entries."""
    log_dir = tmp_path / "logs"
    logger = ExperimentLogger(log_dir=str(log_dir))

    # Log two different types of entries
    logger.log({"metric": "loss", "value": 0.123})
    logger.info("This is a test message.")
    logger.close()

    log_file = log_dir / "log_rank_0.jsonl"
    with open(log_file, "r") as f:
        lines = f.readlines()

    assert len(lines) == 2, "Should have logged two entries."

    # Verify first entry
    entry1 = json.loads(lines[0])
    assert "timestamp" in entry1
    assert entry1["metric"] == "loss"
    assert entry1["value"] == 0.123

    # Verify second entry
    entry2 = json.loads(lines[1])
    assert "timestamp" in entry2
    assert entry2["type"] == "info"
    assert entry2["message"] == "This is a test message."


def test_logger_console_printing(tmp_path: Path, capsys):
    """Verify console printing only happens on the main process and when requested."""
    log_dir = tmp_path / "logs"

    # Case 1: Main process, should print when requested
    logger_main = ExperimentLogger(
        log_dir=str(log_dir), rank=0, is_main_process=True
    )
    logger_main.log({"test": 1}, print_to_console=True)
    captured = capsys.readouterr()
    assert "[LOG]" in captured.out and '"test": 1' in captured.out

    logger_main.log({"test": 2}, print_to_console=False)
    captured = capsys.readouterr()
    assert captured.out == ""
    logger_main.close()

    # Case 2: Non-main process, should never print
    logger_worker = ExperimentLogger(
        log_dir=str(log_dir), rank=1, is_main_process=False
    )
    logger_worker.log({"test": 3}, print_to_console=True)
    captured = capsys.readouterr()
    assert captured.out == ""
    logger_worker.close()
