import pytest

from CLIs.run_multi import validate_config, format_summary, execute_multi_run


@pytest.mark.parametrize(
    "config,expected_missing",
    [
        ({"command": "python main.py", "parameters": {"lr": [0.1, 0.01]}}, []),
        ({"command": "python main.py"}, ["parameters"]),
        ({"parameters": {"lr": [0.1, 0.01]}}, ["command"]),
        ({}, ["command", "parameters"]),
        ({"command": "python main.py", "parameters": {}, "extra": "field"}, []),
    ],
)
def test_validate_config(config, expected_missing):
    """Tests configuration validation with various inputs."""
    missing = validate_config(config)
    assert sorted(missing) == sorted(expected_missing)


def test_validate_config_all_fields_present():
    """Tests validation passes with all required fields."""
    config = {
        "command": "python main.py",
        "parameters": {"lr": [0.1, 0.01], "batch_size": [32, 64]},
        "execution": {"mode": "parallel"},
    }
    missing = validate_config(config)
    assert missing == []


@pytest.mark.parametrize(
    "summary,expected_substrings",
    [
        (
            {"name": "test_run", "total_runs": 10, "successful": 10, "failed": 0},
            ["test_run", "Total runs: 10", "Successful: 10", "Failed: 0"],
        ),
        (
            {"name": "exp_batch", "total_runs": 5, "successful": 3, "failed": 2},
            ["exp_batch", "Total runs: 5", "Successful: 3", "Failed: 2"],
        ),
        (
            {"name": "single", "total_runs": 1, "successful": 0, "failed": 1},
            ["single", "Total runs: 1", "Successful: 0", "Failed: 1"],
        ),
    ],
)
def test_format_summary(summary, expected_substrings):
    """Tests summary formatting with various inputs."""
    formatted = format_summary(summary)
    
    for expected in expected_substrings:
        assert expected in formatted
    
    # Check that it contains separator lines
    assert "=" * 60 in formatted


def test_format_summary_structure():
    """Tests the structure and ordering of formatted summary."""
    summary = {
        "name": "test",
        "total_runs": 5,
        "successful": 4,
        "failed": 1,
    }
    
    formatted = format_summary(summary)
    lines = [line for line in formatted.split("\n") if line]
    
    # Should have 6 lines (empty, separator, name, total, successful, failed, separator)
    assert len(lines) == 6
    
    # Check order
    assert "test" in lines[1]
    assert "Total runs:" in lines[2]
    assert "Successful:" in lines[3]
    assert "Failed:" in lines[4]


def test_execute_multi_run_missing_fields(monkeypatch):
    """Tests that execute_multi_run returns 1 when config is invalid."""
    config = {"command": "python main.py"}  # Missing 'parameters'
    
    # Mock run_multi to ensure it's not called
    def mock_run_multi(config):
        raise AssertionError("run_multi should not be called with invalid config")
    
    monkeypatch.setattr("CLIs.run_multi.run_multi", mock_run_multi)
    
    exit_code = execute_multi_run(config)
    assert exit_code == 1


def test_execute_multi_run_success(monkeypatch, capsys):
    """Tests successful multi-run execution."""
    config = {
        "command": "python main.py",
        "parameters": {"lr": [0.1, 0.01]},
    }
    
    def mock_run_multi(config):
        return {
            "name": "test_run",
            "total_runs": 2,
            "successful": 2,
            "failed": 0,
        }
    
    monkeypatch.setattr("CLIs.run_multi.run_multi", mock_run_multi)
    
    exit_code = execute_multi_run(config)
    assert exit_code == 0
    
    # Check that summary was printed
    captured = capsys.readouterr()
    assert "test_run" in captured.out
    assert "Successful: 2" in captured.out


def test_execute_multi_run_with_failures(monkeypatch, capsys):
    """Tests multi-run execution with some failures."""
    config = {
        "command": "python main.py",
        "parameters": {"lr": [0.1, 0.01]},
    }
    
    def mock_run_multi(config):
        return {
            "name": "test_run",
            "total_runs": 2,
            "successful": 1,
            "failed": 1,
        }
    
    monkeypatch.setattr("CLIs.run_multi.run_multi", mock_run_multi)
    
    exit_code = execute_multi_run(config)
    assert exit_code == 1
    
    # Check that summary was printed
    captured = capsys.readouterr()
    assert "Failed: 1" in captured.out


def test_execute_multi_run_exception(monkeypatch):
    """Tests that exceptions in run_multi are caught and return error code."""
    config = {
        "command": "python main.py",
        "parameters": {"lr": [0.1, 0.01]},
    }
    
    def mock_run_multi(config):
        raise RuntimeError("Something went wrong")
    
    monkeypatch.setattr("CLIs.run_multi.run_multi", mock_run_multi)
    
    exit_code = execute_multi_run(config)
    assert exit_code == 1


@pytest.mark.parametrize(
    "failed_count,expected_exit_code",
    [
        (0, 0),
        (1, 1),
        (5, 1),
        (10, 1),
    ],
)
def test_execute_multi_run_exit_codes(monkeypatch, failed_count, expected_exit_code):
    """Tests that exit code is determined by failure count."""
    config = {
        "command": "python main.py",
        "parameters": {"lr": [0.1, 0.01]},
    }
    
    def mock_run_multi(config):
        return {
            "name": "test",
            "total_runs": 10,
            "successful": 10 - failed_count,
            "failed": failed_count,
        }
    
    monkeypatch.setattr("CLIs.run_multi.run_multi", mock_run_multi)
    
    exit_code = execute_multi_run(config)
    assert exit_code == expected_exit_code

