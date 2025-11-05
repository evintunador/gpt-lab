from unittest import mock

import pytest

from gpt_lab.multi_runner import (
    _expand_parameters,
    _build_command,
    _execute_command,
    run_multi,
)


def test_expand_single_list_parameter():
    """Test parameter expansion with one sweep parameter."""
    params = {"static": 42, "sweep": [1, 2, 3]}
    result = _expand_parameters(params)
    
    assert len(result) == 3
    assert result[0] == {"static": 42, "sweep": 1}
    assert result[1] == {"static": 42, "sweep": 2}
    assert result[2] == {"static": 42, "sweep": 3}


def test_expand_multiple_list_parameters():
    """Test grid expansion with multiple sweep parameters."""
    params = {
        "a": [1, 2],
        "b": [10, 20],
        "c": "constant"
    }
    result = _expand_parameters(params)
    
    # Should produce 2 * 2 = 4 combinations
    assert len(result) == 4
    expected = [
        {"a": 1, "b": 10, "c": "constant"},
        {"a": 1, "b": 20, "c": "constant"},
        {"a": 2, "b": 10, "c": "constant"},
        {"a": 2, "b": 20, "c": "constant"},
    ]
    assert result == expected


def test_expand_three_way_grid():
    """Test expansion with three sweep parameters for typical hyperparameter search."""
    params = {
        "lr": [0.1, 0.01],
        "wd": [0.0, 0.1],
        "batch": [32, 64],
    }
    result = _expand_parameters(params)
    
    # Should produce 2 * 2 * 2 = 8 combinations
    assert len(result) == 8
    assert {"lr": 0.1, "wd": 0.0, "batch": 32} in result
    assert {"lr": 0.01, "wd": 0.1, "batch": 64} in result


@pytest.mark.parametrize(
    "cmd_type,cmd_config,params,expected_start,expected_parts",
    [
        (
            "python",
            {"type": "python", "script": "train.py"},
            {"config": "config.yaml", "lr": 0.001},
            "python train.py",
            ["--config config.yaml", "--lr 0.001"]
        ),
        (
            "torchrun",
            {"type": "torchrun", "script": "train.py", "nproc_per_node": 4},
            {"batch_size": 32},
            "torchrun --nproc_per_node=4",
            ["train.py", "--batch_size 32"]
        ),
        (
            "sbatch",
            {"type": "sbatch", "script": "job.py", "sbatch_flags": "--nodes=2"},
            {"epochs": 10},
            "sbatch --nodes=2",
            ["job.py", "--epochs 10"]
        ),
    ],
)
def test_build_command_types(cmd_type, cmd_config, params, expected_start, expected_parts):
    """Test building commands for different execution types."""
    result = _build_command(cmd_config, params)
    
    assert result.startswith(expected_start)
    for part in expected_parts:
        assert part in result


def test_build_command_with_boolean_flags():
    """Test that boolean parameters are handled correctly in commands."""
    command_config = {"type": "python", "script": "test.py"}
    params = {"flag": True, "no_flag": False, "value": 42}
    
    result = _build_command(command_config, params)
    
    # True should add the flag, False should be omitted
    assert "--flag" in result
    assert "--no_flag" not in result
    assert "--value 42" in result


def test_build_command_with_nested_parameters():
    """Test that nested parameter keys with dots are preserved."""
    command_config = {"type": "python", "script": "train.py"}
    params = {
        "model.dim": 512,
        "training.lr": 0.001
    }
    
    result = _build_command(command_config, params)
    
    assert "--model.dim 512" in result
    assert "--training.lr 0.001" in result


def test_build_torchrun_command_with_distributed_options():
    """Test building torchrun command with full distributed configuration."""
    command_config = {
        "type": "torchrun",
        "script": "train.py",
        "nproc_per_node": 2,
        "nnodes": 4,
        "node_rank": 1,
        "master_addr": "192.168.1.1",
        "master_port": 29500
    }
    params = {}
    
    result = _build_command(command_config, params)
    
    assert "--nproc_per_node=2" in result
    assert "--nnodes=4" in result
    assert "--node_rank=1" in result
    assert "--master_addr=192.168.1.1" in result
    assert "--master_port=29500" in result


def test_build_command_unsupported_type_raises_error():
    """Test that unsupported command types raise appropriate errors."""
    command_config = {"type": "unsupported", "script": "test.py"}
    params = {}
    
    with pytest.raises(ValueError, match="Unsupported command type"):
        _build_command(command_config, params)


@mock.patch('subprocess.run')
def test_execute_command_success(mock_run):
    """Test successful command execution."""
    mock_run.return_value = mock.Mock(
        returncode=0,
        stdout="Success output",
        stderr=""
    )
    
    name, code, stdout, stderr = _execute_command("echo test", "test_run")
    
    assert name == "test_run"
    assert code == 0
    assert stdout == "Success output"
    assert stderr == ""
    mock_run.assert_called_once()


@mock.patch('subprocess.run')
def test_execute_command_failure(mock_run):
    """Test failed command execution with non-zero exit code."""
    mock_run.return_value = mock.Mock(
        returncode=1,
        stdout="",
        stderr="Error message"
    )
    
    name, code, stdout, stderr = _execute_command("false", "failing_run")
    
    assert name == "failing_run"
    assert code == 1
    assert stderr == "Error message"


@mock.patch('subprocess.run')
def test_execute_command_exception_handling(mock_run):
    """Test that exceptions during execution are caught and reported."""
    mock_run.side_effect = Exception("Unexpected error")
    
    name, code, stdout, stderr = _execute_command("bad command", "error_run")
    
    assert name == "error_run"
    assert code == -1
    assert stdout == ""
    assert "Unexpected error" in stderr


@mock.patch('gpt_lab.multi_runner._execute_command')
def test_run_multi_sequential_execution(mock_execute):
    """Test running multiple experiments in sequential mode."""
    mock_execute.return_value = ("run", 0, "", "")
    
    config = {
        "name": "test_sweep",
        "command": {"type": "python", "script": "test.py"},
        "parameters": {"lr": [0.1, 0.01]},
        "execution": {"mode": "sequential"}
    }
    
    summary = run_multi(config)
    
    assert summary["name"] == "test_sweep"
    assert summary["total_runs"] == 2
    assert summary["successful"] == 2
    assert summary["failed"] == 0
    assert mock_execute.call_count == 2


@mock.patch('gpt_lab.multi_runner._execute_command')
@mock.patch('gpt_lab.multi_runner.ProcessPoolExecutor')
def test_run_multi_parallel_execution(mock_executor_class, mock_execute):
    """Test running multiple experiments in parallel mode."""
    # Mock the executor to avoid actual multiprocessing
    mock_executor = mock.MagicMock()
    mock_executor_class.return_value.__enter__.return_value = mock_executor
    
    # Create mock futures
    mock_futures = []
    for i in range(4):
        mock_future = mock.MagicMock()
        mock_future.result.return_value = (f"run_{i}", 0, "", "")
        mock_futures.append(mock_future)
    
    mock_executor.submit.side_effect = mock_futures
    
    with mock.patch('gpt_lab.multi_runner.as_completed', return_value=mock_futures):
        config = {
            "name": "parallel_sweep",
            "command": {"type": "python", "script": "test.py"},
            "parameters": {"value": [1, 2, 3, 4]},
            "execution": {"mode": "parallel", "max_workers": 2}
        }
        
        summary = run_multi(config)
        
        assert summary["total_runs"] == 4
        assert summary["successful"] == 4
        mock_executor_class.assert_called_once_with(max_workers=2)
        assert mock_executor.submit.call_count == 4


@mock.patch('gpt_lab.multi_runner._execute_command')
def test_run_multi_with_failures(mock_execute):
    """Test that run_multi correctly tracks both successes and failures."""
    mock_execute.side_effect = [
        ("run1", 0, "", ""),
        ("run2", 0, "", ""),
        ("run3", 1, "", "error"),
    ]
    
    config = {
        "name": "mixed_results",
        "command": {"type": "python", "script": "test.py"},
        "parameters": {"idx": [1, 2, 3]},
        "execution": {"mode": "sequential"}
    }
    
    summary = run_multi(config)
    
    assert summary["total_runs"] == 3
    assert summary["successful"] == 2
    assert summary["failed"] == 1


@mock.patch('gpt_lab.multi_runner._execute_command')
def test_run_multi_grid_expansion(mock_execute):
    """Test that run_multi correctly expands parameter grids."""
    mock_execute.return_value = ("run", 0, "", "")
    
    config = {
        "name": "grid_test",
        "command": {"type": "python", "script": "train.py"},
        "parameters": {
            "lr": [0.1, 0.01],
            "wd": [0.0, 0.1],
            "batch": 32  # Static parameter
        },
        "execution": {"mode": "sequential"}
    }
    
    summary = run_multi(config)
    
    # Should generate 2 * 2 = 4 runs
    assert summary["total_runs"] == 4
    assert mock_execute.call_count == 4
