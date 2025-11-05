import subprocess
from pathlib import Path

import pytest

from CLIs.scaffold_experiment import scaffold_experiment, TEMPLATE_YAML


@pytest.fixture
def scaffolded_experiment(tmp_path):
    """Runs the scaffold_experiment script and returns the experiment path."""
    exp_name = "my_test_experiment"
    scaffold_experiment(tmp_path, exp_name)
    exp_dir = tmp_path / "experiments" / exp_name
    return exp_dir, exp_name


def test_directory_and_file_structure(scaffolded_experiment):
    """Tests that the correct directories and files are created."""
    exp_dir, _ = scaffolded_experiment
    assert exp_dir.is_dir()
    assert (exp_dir / "gpt_lab").is_dir()
    artifacts_dir = exp_dir / "gpt_lab" / "artifacts"
    assert artifacts_dir.is_dir()
    assert (artifacts_dir / ".gitkeep").exists()
    assert (exp_dir / "gpt_lab.yaml").is_file()
    assert (exp_dir / "main.py").is_file()


def test_file_contents(scaffolded_experiment):
    """Tests that the created files have the correct content."""
    exp_dir, exp_name = scaffolded_experiment

    yaml_path = exp_dir / "gpt_lab.yaml"
    assert yaml_path.read_text() == TEMPLATE_YAML

    main_path = exp_dir / "main.py"
    expected_main_content = f"""if __name__ == '__main__':
    print('Hello from experiment: {exp_name}')
"""
    assert main_path.read_text() == expected_main_content


def test_git_repository_initialization(scaffolded_experiment):
    """Tests that a git repository is initialized."""
    exp_dir, _ = scaffolded_experiment
    assert (exp_dir / ".git").is_dir()


def test_git_repo_is_clean(scaffolded_experiment):
    """Tests that the git repository has no uncommitted changes."""
    exp_dir, _ = scaffolded_experiment
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=exp_dir,
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout == ""


def test_git_initial_commit(scaffolded_experiment):
    """Tests the initial commit message."""
    exp_dir, _ = scaffolded_experiment
    result = subprocess.run(
        ["git", "log", "-1", "--pretty=%B"],
        cwd=exp_dir,
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == "Initial commit from scaffold"
