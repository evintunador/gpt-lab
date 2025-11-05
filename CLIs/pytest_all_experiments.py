import argparse
import os
import subprocess
import sys
from pathlib import Path


def run_one(exp_name: str, extra_pytest_args: list[str], python_executable: str = None) -> int:
    """Run pytest for a single experiment with isolated environment variables.
    
    Args:
        exp_name: Name of the experiment to test
        extra_pytest_args: Additional arguments to pass to pytest
        python_executable: Python executable path (defaults to sys.executable)
        
    Returns:
        Return code from pytest subprocess
    """
    env = os.environ.copy()
    env["GPT_LAB_CURRENT_EXPERIMENT"] = exp_name
    env["GPT_LAB_ACTIVE_EXPERIMENTS"] = exp_name
    # Leave packs empty unless overridden by caller
    
    python_exec = python_executable or sys.executable
    cmd = [python_exec, "-m", "pytest"] + extra_pytest_args
    print(f"\n=== Running pytest for experiment: {exp_name} ===")
    proc = subprocess.run(cmd, env=env)
    return proc.returncode


def find_repo_root(start_path: Path) -> Path:
    """Find repository root by looking for pyproject.toml or .git directory.
    
    Args:
        start_path: Path to start searching from
        
    Returns:
        Path to repository root
        
    Raises:
        RuntimeError: If repository root cannot be found
    """
    current = start_path.resolve()
    while current != current.parent:
        if (current / "pyproject.toml").exists() or (current / ".git").exists():
            return current
        current = current.parent
    raise RuntimeError("Could not find repository root (no pyproject.toml or .git found)")


def get_experiments(repo_root: Path, include: list[str] = None, exclude: list[str] = None) -> list[str]:
    """Get list of experiments to test based on include/exclude filters.
    
    Args:
        repo_root: Path to repository root
        include: List of experiment names to include (None means include all)
        exclude: List of experiment names to exclude (None means exclude none)
        
    Returns:
        List of experiment names to test
    """
    experiments_dir = repo_root / "experiments"
    if not experiments_dir.exists():
        return []
    
    exps = [p.name for p in experiments_dir.iterdir() if p.is_dir()]
    
    if include is not None:
        include_set = set(include)
        exps = [e for e in exps if e in include_set]
    
    if exclude is not None:
        exclude_set = set(exclude)
        exps = [e for e in exps if e not in exclude_set]
    
    return exps


def run_all_experiments(
    repo_root: Path,
    experiments: list[str],
    extra_pytest_args: list[str],
    python_executable: str = None,
) -> list[str]:
    """Run pytest for all specified experiments.
    
    Args:
        repo_root: Path to repository root
        experiments: List of experiment names to test
        extra_pytest_args: Additional arguments to pass to pytest
        python_executable: Python executable path (defaults to sys.executable)
        
    Returns:
        List of experiment names that failed tests
    """
    failures = []
    for exp_name in experiments:
        code = run_one(exp_name, extra_pytest_args, python_executable)
        if code != 0:
            failures.append(exp_name)
    return failures


def main():
    parser = argparse.ArgumentParser(description="Run pytest per experiment in isolation")
    parser.add_argument("--include", nargs="*", default=None, help="Experiment names to include")
    parser.add_argument("--exclude", nargs="*", default=None, help="Experiment names to exclude")
    parser.add_argument("--pytest-args", nargs=argparse.REMAINDER, help="Additional args passed to pytest")
    args = parser.parse_args()

    repo = find_repo_root(Path(__file__))
    exps = get_experiments(repo, include=args.include, exclude=args.exclude)

    extra = args.pytest_args or []
    failures = run_all_experiments(repo, exps, extra)

    if failures:
        print(f"\nExperiments failing tests: {failures}")
        sys.exit(1)
    print("\nAll experiment test runs succeeded.")


if __name__ == "__main__":
    main()


