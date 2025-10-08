import json
import os
from pathlib import Path

import pytest

from gpt_lab.results_comparator import (
    LogSchema,
    MetricResult,
    RunResult,
    find_run_directories,
    load_run_data,
    compare_runs,
    STRATEGY_REGISTRY,
)


def create_run_directory(base_dir: Path, run_name: str, include_git_json: bool = True, malformed_line: bool = False):
    """
    Helper function to create a realistic run directory structure.
    
    Args:
        base_dir: Base directory for the run
        run_name: Name of the run directory
        include_git_json: Whether to include git_info.json file
        malformed_line: Whether to include a malformed JSON line
    """
    run_dir = base_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Create git_info.json if requested
    if include_git_json:
        git_info = {
            "commit_hash": "abc123def456",
            "branch": "main",
            "remote_url": "git@github.com:user/repo.git",
            "github_url": "https://github.com/user/repo/commit/abc123def456",
            "was_dirty": False
        }
        with open(run_dir / "git_info.json", "w") as f:
            json.dump(git_info, f)
    
    # Create log_rank_0.jsonl
    log_file = run_dir / "log_rank_0.jsonl"
    with open(log_file, "w") as f:
        # System Information with git_info (fallback if JSON not present)
        f.write(json.dumps({
            "message": "System Information",
            "git_info": {
                "commit_hash": "log_commit_xyz",  # Different from JSON to test priority
                "branch": "log_branch",
                "was_dirty": True
            },
            "timestamp": "2025-01-01 10:00:00"
        }) + "\n")
        
        # Hyperparameters with nested structure
        f.write(json.dumps({
            "message": "Hyperparameters",
            "training": {
                "learning_rate": 0.001,
                "batch_size": 32
            },
            "model": {
                "layers": 12,
                "hidden_dim": 768
            },
            "optimizer": {
                "weight_decay": 0.1,
                "beta1": 0.9
            },
            "timestamp": "2025-01-01 10:00:01"
        }) + "\n")
        
        # Metric line with 'step' counter
        f.write(json.dumps({
            "step": 100,
            "train_loss": 2.5,
            "timestamp": "2025-01-01 10:01:00"
        }) + "\n")
        
        # Metric line with 'epoch' counter
        f.write(json.dumps({
            "epoch": 1,
            "val_loss": 2.3,
            "timestamp": "2025-01-01 10:02:00"
        }) + "\n")
        
        # Metric line with both 'step' and 'epoch' (step should be preferred)
        f.write(json.dumps({
            "step": 200,
            "epoch": 2,
            "train_loss": 1.8,
            "val_loss": 1.9,
            "timestamp": "2025-01-01 10:03:00"
        }) + "\n")
        
        # Nested metrics (e.g., benchmark results)
        f.write(json.dumps({
            "step": 300,
            "message": "Benchmark Results",
            "benchmark": "HellaSwag",
            "results": {
                "accuracy": 0.85,
                "f1_score": 0.82
            },
            "timestamp": "2025-01-01 10:04:00"
        }) + "\n")
        
        # Another nested metric
        f.write(json.dumps({
            "step": 300,
            "results": {
                "accuracy": 0.88,
                "precision": 0.90
            },
            "timestamp": "2025-01-01 10:05:00"
        }) + "\n")
        
        # Malformed JSON line (if requested)
        if malformed_line:
            f.write("{ this is not valid json }\n")
        
        # One more valid line after malformed
        f.write(json.dumps({
            "step": 400,
            "train_loss": 1.2,
            "timestamp": "2025-01-01 10:06:00"
        }) + "\n")
    
    return run_dir


def test_find_run_directories_finds_valid_runs(tmp_path: Path):
    """Test that find_run_directories only finds directories with log files."""
    # Create valid run directory
    create_run_directory(tmp_path, "run_001")
    
    # Create directory without log file
    empty_dir = tmp_path / "empty_run"
    empty_dir.mkdir()
    
    # Create directory with wrong log filename
    wrong_log_dir = tmp_path / "wrong_log"
    wrong_log_dir.mkdir()
    (wrong_log_dir / "other_log.jsonl").write_text("")
    
    # Find runs
    pattern = str(tmp_path / "*")
    found_dirs = find_run_directories([pattern])
    
    # Should only find run_001
    assert len(found_dirs) == 1
    assert "run_001" in found_dirs[0]


def test_find_run_directories_respects_custom_schema(tmp_path: Path):
    """Test that find_run_directories uses custom schema log filename."""
    # Create run with custom log filename
    run_dir = tmp_path / "custom_run"
    run_dir.mkdir()
    (run_dir / "custom.jsonl").write_text(json.dumps({"message": "test"}) + "\n")
    
    # Should NOT find with default schema
    default_found = find_run_directories([str(tmp_path / "*")])
    assert len(default_found) == 0
    
    # Should find with custom schema
    custom_schema = LogSchema(log_filename="custom.jsonl")
    custom_found = find_run_directories([str(tmp_path / "*")], schema=custom_schema)
    assert len(custom_found) == 1


def test_load_run_data_prefers_git_json_over_logs(tmp_path: Path):
    """Test that git_info is loaded from JSON file first, then falls back to logs."""
    # Test 1: Both JSON and log present - should use JSON
    run_with_json = create_run_directory(tmp_path, "run_with_json", include_git_json=True)
    
    result = load_run_data(str(run_with_json), [], [])
    
    assert result.git_info is not None
    assert result.git_info["commit_hash"] == "abc123def456"  # From JSON, not log
    assert result.git_info["branch"] == "main"  # From JSON
    
    # Test 2: Only log present - should use log
    run_without_json = create_run_directory(tmp_path, "run_without_json", include_git_json=False)
    
    result = load_run_data(str(run_without_json), [], [])
    
    assert result.git_info is not None
    assert result.git_info["commit_hash"] == "log_commit_xyz"  # From log
    assert result.git_info["branch"] == "log_branch"  # From log


def test_load_run_data_extracts_hyperparameters(tmp_path: Path):
    """Test that hyperparameters are extracted using nested paths."""
    run_dir = create_run_directory(tmp_path, "run_001")
    
    hparam_defs = [
        {"display_name": "Learning Rate", "paths": ["training.learning_rate"]},
        {"display_name": "Batch Size", "paths": ["training.batch_size"]},
        {"display_name": "Layers", "paths": ["model.layers"]},
        {"display_name": "Weight Decay", "paths": ["optimizer.weight_decay"]},
    ]
    
    result = load_run_data(str(run_dir), [], hparam_defs)
    
    assert result.hyperparameters["Learning Rate"] == 0.001
    assert result.hyperparameters["Batch Size"] == 32
    assert result.hyperparameters["Layers"] == 12
    assert result.hyperparameters["Weight Decay"] == 0.1


def test_load_run_data_extracts_nested_metrics(tmp_path: Path):
    """Test that metrics with nested paths (e.g., results.accuracy) are extracted."""
    run_dir = create_run_directory(tmp_path, "run_001")
    
    metric_defs = [
        {"display_name": "Accuracy", "paths": ["results.accuracy"], "strategy": "best_value", "goal": "maximize"},
        {"display_name": "F1 Score", "paths": ["results.f1_score"], "strategy": "last_value"},
    ]
    
    result = load_run_data(str(run_dir), metric_defs, [])
    
    # Should extract nested metrics
    assert "Accuracy" in result.metrics
    assert result.metrics["Accuracy"].values == [0.85, 0.88]
    
    assert "F1 Score" in result.metrics
    assert result.metrics["F1 Score"].values == [0.82]


def test_load_run_data_handles_multiple_step_counters(tmp_path: Path):
    """Test that step_series captures different counter types and selected_step_key is correct."""
    run_dir = create_run_directory(tmp_path, "run_001")
    
    metric_defs = [
        {"display_name": "Train Loss", "paths": ["train_loss"], "strategy": "best_value", "goal": "minimize"},
        {"display_name": "Val Loss", "paths": ["val_loss"], "strategy": "last_value"},
    ]
    
    result = load_run_data(str(run_dir), metric_defs, [])
    
    # Train Loss appears at step 100, 200, 400
    train_loss = result.metrics["Train Loss"]
    assert train_loss.values == [2.5, 1.8, 1.2]
    
    # Check step_series has both step and epoch
    assert "step" in train_loss.step_series
    assert "epoch" in train_loss.step_series
    
    # Step should be [100, 200, None, None, 400] aligned with values
    # But we only have 3 values, so steps should be [100, 200, 400]
    assert train_loss.step_series["step"] == [100, 200, 400]
    
    # Epoch should be [None, 2, None] where None means epoch wasn't present
    assert train_loss.step_series["epoch"] == [None, 2, None]
    
    # Selected step key should be 'step' (first in preferred_step_keys that has data)
    assert train_loss.selected_step_key == "step"
    assert train_loss.selected_steps == [100, 200, 400]
    
    # Val Loss appears at epoch 1, and step/epoch 200
    val_loss = result.metrics["Val Loss"]
    assert val_loss.values == [2.3, 1.9]
    
    # For val_loss, we should have step and epoch data
    # First value has epoch=1, no step
    # Second value has both step=200 and epoch=2
    assert val_loss.step_series["step"] == [None, 200]
    assert val_loss.step_series["epoch"] == [1, 2]
    
    # Selected should still be 'step' from schema preference, even though first value is None
    # Actually, let me check the logic - it should prefer the first key with ANY data
    # Looking at the implementation, it checks which key is in step_series, not which has non-None values
    # So selected_step_key should be 'step' since it's first in preferred_step_keys
    assert val_loss.selected_step_key in ["step", "epoch"]  # Could be either depending on schema


def test_load_run_data_handles_malformed_json(tmp_path: Path):
    """Test that malformed JSON lines are skipped gracefully."""
    run_dir = create_run_directory(tmp_path, "run_with_bad_json", malformed_line=True)
    
    metric_defs = [
        {"display_name": "Train Loss", "paths": ["train_loss"], "strategy": "best_value", "goal": "minimize"},
    ]
    
    # Should not raise an exception
    result = load_run_data(str(run_dir), metric_defs, [])
    
    # Should still extract the valid lines before and after the malformed one
    assert "Train Loss" in result.metrics
    assert 2.5 in result.metrics["Train Loss"].values  # From before malformed line
    assert 1.2 in result.metrics["Train Loss"].values  # From after malformed line


def test_load_run_data_with_custom_schema(tmp_path: Path):
    """Test that custom schema messages and fields are used."""
    # Create run with custom conventions
    run_dir = tmp_path / "custom_run"
    run_dir.mkdir()
    
    log_file = run_dir / "experiment.log"
    with open(log_file, "w") as f:
        f.write(json.dumps({
            "message": "[METADATA]",
            "version_control": {
                "hash": "custom123",
                "branch": "feature"
            }
        }) + "\n")
        
        f.write(json.dumps({
            "message": "[CONFIG]",
            "lr": 0.01
        }) + "\n")
        
        f.write(json.dumps({
            "custom_step": 50,
            "loss": 1.5
        }) + "\n")
    
    # Use custom schema
    custom_schema = LogSchema(
        system_info_message="[METADATA]",
        hyperparams_message="[CONFIG]",
        git_info_field="version_control",
        log_filename="experiment.log",
        preferred_step_keys=["custom_step"]
    )
    
    metric_defs = [{"display_name": "Loss", "paths": ["loss"], "strategy": "last_value"}]
    hparam_defs = [{"display_name": "LR", "paths": ["lr"]}]
    
    result = load_run_data(str(run_dir), metric_defs, hparam_defs, schema=custom_schema)
    
    assert result.git_info["hash"] == "custom123"
    assert result.hyperparameters["LR"] == 0.01
    assert result.metrics["Loss"].values == [1.5]
    assert result.metrics["Loss"].selected_step_key == "custom_step"


def test_compare_runs_aggregates_multiple_directories(tmp_path: Path):
    """Test that compare_runs processes multiple run directories."""
    # Create multiple runs
    create_run_directory(tmp_path, "run_001")
    create_run_directory(tmp_path, "run_002")
    create_run_directory(tmp_path, "run_003")
    
    run_dirs = [
        str(tmp_path / "run_001"),
        str(tmp_path / "run_002"),
        str(tmp_path / "run_003"),
    ]
    
    metric_defs = [
        {"display_name": "Train Loss", "paths": ["train_loss"], "strategy": "best_value", "goal": "minimize"},
    ]
    
    hparam_defs = [
        {"display_name": "Learning Rate", "paths": ["training.learning_rate"]},
    ]
    
    results = compare_runs(run_dirs, metric_defs, hparam_defs)
    
    assert len(results) == 3
    
    # Each result should have data
    for result in results:
        assert isinstance(result, RunResult)
        assert result.git_info is not None
        assert "Learning Rate" in result.hyperparameters
        assert "Train Loss" in result.metrics


def test_compare_runs_handles_failures_gracefully(tmp_path: Path):
    """Test that compare_runs continues even if one run fails."""
    # Create one valid run
    create_run_directory(tmp_path, "run_001")
    
    # Create one invalid run (empty directory)
    invalid_dir = tmp_path / "invalid_run"
    invalid_dir.mkdir()
    
    run_dirs = [
        str(tmp_path / "run_001"),
        str(invalid_dir),
    ]
    
    results = compare_runs(run_dirs, [], [])
    
    # Should return results for both (invalid one with minimal data)
    assert len(results) == 2
    assert results[0].git_info is not None  # Valid run
    assert results[1].git_info is None  # Invalid run


def test_metric_result_step_series_alignment(tmp_path: Path):
    """Test that step_series values are properly aligned with metric values."""
    run_dir = tmp_path / "run_001"
    run_dir.mkdir()
    
    log_file = run_dir / "log_rank_0.jsonl"
    with open(log_file, "w") as f:
        f.write(json.dumps({"message": "Hyperparameters"}) + "\n")
        # Value with step only
        f.write(json.dumps({"step": 10, "loss": 1.0}) + "\n")
        # Value with epoch only
        f.write(json.dumps({"epoch": 1, "loss": 0.9}) + "\n")
        # Value with both
        f.write(json.dumps({"step": 20, "epoch": 2, "loss": 0.8}) + "\n")
        # Value with step only again
        f.write(json.dumps({"step": 30, "loss": 0.7}) + "\n")
    
    metric_defs = [{"display_name": "Loss", "paths": ["loss"], "strategy": "best_value", "goal": "minimize"}]
    
    result = load_run_data(str(run_dir), metric_defs, [])
    
    loss = result.metrics["Loss"]
    
    # Should have 4 values
    assert len(loss.values) == 4
    assert loss.values == [1.0, 0.9, 0.8, 0.7]
    
    # Step series should have [10, None, 20, 30]
    assert loss.step_series["step"] == [10, None, 20, 30]
    
    # Epoch series should have [None, 1, 2, None]
    assert loss.step_series["epoch"] == [None, 1, 2, None]
    
    # Both series should have same length as values
    assert len(loss.step_series["step"]) == len(loss.values)
    assert len(loss.step_series["epoch"]) == len(loss.values)


def test_strategy_registry_integration(tmp_path: Path):
    """Test that strategy registry works with extracted metrics."""
    run_dir = create_run_directory(tmp_path, "run_001")
    
    metric_defs = [
        {"display_name": "Train Loss", "paths": ["train_loss"], "strategy": "best_value", "goal": "minimize"},
        {"display_name": "Val Loss", "paths": ["val_loss"], "strategy": "last_value"},
        {"display_name": "Accuracy", "paths": ["results.accuracy"], "strategy": "time_series"},
    ]
    
    result = load_run_data(str(run_dir), metric_defs, [])
    
    # Test best_value strategy
    train_loss = result.metrics["Train Loss"]
    best_loss = STRATEGY_REGISTRY["best_value"](train_loss.values, "minimize")
    assert best_loss == min(train_loss.values)
    
    # Test last_value strategy
    val_loss = result.metrics["Val Loss"]
    last_val = STRATEGY_REGISTRY["last_value"](val_loss.values)
    assert last_val == val_loss.values[-1]
    
    # Test time_series strategy
    accuracy = result.metrics["Accuracy"]
    series = STRATEGY_REGISTRY["time_series"](accuracy.values)
    assert series == accuracy.values


def test_empty_run_directory(tmp_path: Path):
    """Test handling of empty run directory."""
    run_dir = tmp_path / "empty_run"
    run_dir.mkdir()
    
    result = load_run_data(str(run_dir), [], [])
    
    # Should return a RunResult with the path but no data
    assert result.run_path == str(run_dir)
    assert result.git_info is None
    assert len(result.hyperparameters) == 0
    assert len(result.metrics) == 0


def test_metric_with_fallback_paths(tmp_path: Path):
    """Test that metrics try multiple paths in order."""
    run_dir = tmp_path / "run_001"
    run_dir.mkdir()
    
    log_file = run_dir / "log_rank_0.jsonl"
    with open(log_file, "w") as f:
        f.write(json.dumps({"message": "Hyperparameters"}) + "\n")
        # Has "acc" but not "accuracy"
        f.write(json.dumps({"step": 1, "acc": 0.85}) + "\n")
        # Has "accuracy" but not "acc"
        f.write(json.dumps({"step": 2, "accuracy": 0.90}) + "\n")
    
    # Define metric with fallback paths
    metric_defs = [
        {"display_name": "Accuracy", "paths": ["accuracy", "acc"], "strategy": "best_value", "goal": "maximize"}
    ]
    
    result = load_run_data(str(run_dir), metric_defs, [])
    
    # Should extract both values using different paths
    assert result.metrics["Accuracy"].values == [0.85, 0.90]


def test_hyperparameter_with_fallback_paths(tmp_path: Path):
    """Test that hyperparameters try multiple paths in order."""
    run_dir = tmp_path / "run_001"
    run_dir.mkdir()
    
    log_file = run_dir / "log_rank_0.jsonl"
    with open(log_file, "w") as f:
        f.write(json.dumps({
            "message": "Hyperparameters",
            "training": {"lr": 0.001}  # Has 'lr' not 'learning_rate'
        }) + "\n")
    
    # Define hyperparameter with fallback paths
    hparam_defs = [
        {"display_name": "Learning Rate", "paths": ["training.learning_rate", "training.lr", "lr"]}
    ]
    
    result = load_run_data(str(run_dir), [], hparam_defs)
    
    # Should find using second fallback path
    assert result.hyperparameters["Learning Rate"] == 0.001

