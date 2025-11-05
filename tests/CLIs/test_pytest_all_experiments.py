from pathlib import Path

import pytest

from CLIs.pytest_all_experiments import find_repo_root, get_experiments, run_all_experiments


@pytest.mark.parametrize(
    "has_pyproject,has_git,should_find",
    [
        (True, False, True),
        (False, True, True),
        (True, True, True),
        (False, False, False),
    ],
)
def test_find_repo_root(tmp_path, has_pyproject, has_git, should_find):
    """Tests finding repository root with various markers."""
    if has_pyproject:
        (tmp_path / "pyproject.toml").touch()
    if has_git:
        (tmp_path / ".git").mkdir()

    # Create a subdirectory to search from
    subdir = tmp_path / "sub" / "directory"
    subdir.mkdir(parents=True)

    if should_find:
        result = find_repo_root(subdir)
        assert result == tmp_path
    else:
        with pytest.raises(RuntimeError, match="Could not find repository root"):
            find_repo_root(subdir)


def test_find_repo_root_from_root():
    """Tests finding repo root when starting at the root itself."""
    start_path = Path(__file__).resolve()
    # Find the actual repo root
    repo = find_repo_root(start_path)
    
    # Should be able to find it again from the root
    result = find_repo_root(repo)
    assert result == repo


@pytest.mark.parametrize(
    "exp_names,include,exclude,expected",
    [
        (["exp1", "exp2", "exp3"], None, None, ["exp1", "exp2", "exp3"]),
        (["exp1", "exp2", "exp3"], ["exp1"], None, ["exp1"]),
        (["exp1", "exp2", "exp3"], ["exp1", "exp3"], None, ["exp1", "exp3"]),
        (["exp1", "exp2", "exp3"], None, ["exp2"], ["exp1", "exp3"]),
        (["exp1", "exp2", "exp3"], None, ["exp1", "exp3"], ["exp2"]),
        (["exp1", "exp2", "exp3"], ["exp1", "exp2"], ["exp1"], ["exp2"]),
        (["exp1", "exp2", "exp3"], ["nonexistent"], None, []),
        (["exp1", "exp2", "exp3"], None, ["exp1", "exp2", "exp3"], []),
        ([], None, None, []),
    ],
)
def test_get_experiments(tmp_path, exp_names, include, exclude, expected):
    """Tests filtering experiments with include/exclude lists."""
    experiments_dir = tmp_path / "experiments"
    experiments_dir.mkdir()

    # Create experiment directories
    for exp_name in exp_names:
        (experiments_dir / exp_name).mkdir()

    result = get_experiments(tmp_path, include=include, exclude=exclude)
    
    # Sort both lists for comparison since order may vary
    assert sorted(result) == sorted(expected)


def test_get_experiments_no_experiments_dir(tmp_path):
    """Tests behavior when experiments directory doesn't exist."""
    result = get_experiments(tmp_path, include=None, exclude=None)
    assert result == []


def test_get_experiments_with_files(tmp_path):
    """Tests that files in experiments directory are ignored."""
    experiments_dir = tmp_path / "experiments"
    experiments_dir.mkdir()

    # Create some directories and files
    (experiments_dir / "exp1").mkdir()
    (experiments_dir / "exp2").mkdir()
    (experiments_dir / "not_a_dir.txt").write_text("ignore me")

    result = get_experiments(tmp_path, include=None, exclude=None)
    assert sorted(result) == ["exp1", "exp2"]


@pytest.mark.parametrize(
    "experiments,expected_failures",
    [
        ([], []),
        (["exp1"], []),
        (["exp1", "exp2"], []),
    ],
)
def test_run_all_experiments_success(tmp_path, monkeypatch, experiments, expected_failures):
    """Tests running all experiments when all succeed."""
    # Mock run_one to always return 0 (success)
    def mock_run_one(exp_name, extra_args, python_exec=None):
        return 0
    
    monkeypatch.setattr("CLIs.pytest_all_experiments.run_one", mock_run_one)
    
    failures = run_all_experiments(tmp_path, experiments, [])
    assert failures == expected_failures


def test_run_all_experiments_with_failures(tmp_path, monkeypatch):
    """Tests running all experiments when some fail."""
    # Mock run_one to fail for specific experiments
    def mock_run_one(exp_name, extra_args, python_exec=None):
        if exp_name in ["exp2", "exp4"]:
            return 1  # Failure
        return 0  # Success
    
    monkeypatch.setattr("CLIs.pytest_all_experiments.run_one", mock_run_one)
    
    experiments = ["exp1", "exp2", "exp3", "exp4"]
    failures = run_all_experiments(tmp_path, experiments, [])
    assert sorted(failures) == ["exp2", "exp4"]


def test_run_all_experiments_all_fail(tmp_path, monkeypatch):
    """Tests running all experiments when all fail."""
    # Mock run_one to always return 1 (failure)
    def mock_run_one(exp_name, extra_args, python_exec=None):
        return 1
    
    monkeypatch.setattr("CLIs.pytest_all_experiments.run_one", mock_run_one)
    
    experiments = ["exp1", "exp2", "exp3"]
    failures = run_all_experiments(tmp_path, experiments, [])
    assert sorted(failures) == ["exp1", "exp2", "exp3"]


def test_run_all_experiments_passes_pytest_args(tmp_path, monkeypatch):
    """Tests that extra pytest args are passed to run_one."""
    called_with = []
    
    def mock_run_one(exp_name, extra_args, python_exec=None):
        called_with.append((exp_name, extra_args, python_exec))
        return 0
    
    monkeypatch.setattr("CLIs.pytest_all_experiments.run_one", mock_run_one)
    
    experiments = ["exp1", "exp2"]
    extra_args = ["-v", "--tb=short"]
    python_exec = "/usr/bin/python3"
    
    run_all_experiments(tmp_path, experiments, extra_args, python_exec)
    
    assert len(called_with) == 2
    assert called_with[0] == ("exp1", extra_args, python_exec)
    assert called_with[1] == ("exp2", extra_args, python_exec)

