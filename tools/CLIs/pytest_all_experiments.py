import argparse
import os
import subprocess
import sys
from pathlib import Path


def run_one(exp_name: str, extra_pytest_args: list[str]) -> int:
    env = os.environ.copy()
    env["GPT_LAB_CURRENT_EXPERIMENT"] = exp_name
    env["GPT_LAB_ACTIVE_EXPERIMENTS"] = exp_name
    # Leave packs empty unless overridden by caller
    cmd = [sys.executable, "-m", "pytest"] + extra_pytest_args
    print(f"\n=== Running pytest for experiment: {exp_name} ===")
    proc = subprocess.run(cmd, env=env)
    return proc.returncode


def main():
    parser = argparse.ArgumentParser(description="Run pytest per experiment in isolation")
    parser.add_argument("--include", nargs="*", default=None, help="Experiment names to include")
    parser.add_argument("--exclude", nargs="*", default=None, help="Experiment names to exclude")
    parser.add_argument("--pytest-args", nargs=argparse.REMAINDER, help="Additional args passed to pytest")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[2]
    exps = [p.name for p in (repo / "experiments").iterdir() if p.is_dir()]
    if args.include:
        exps = [e for e in exps if e in set(args.include)]
    if args.exclude:
        excl = set(args.exclude)
        exps = [e for e in exps if e not in excl]

    extra = args.pytest_args or []
    failures = []
    for e in exps:
        code = run_one(e, extra)
        if code != 0:
            failures.append(e)

    if failures:
        print(f"\nExperiments failing tests: {failures}")
        sys.exit(1)
    print("\nAll experiment test runs succeeded.")


if __name__ == "__main__":
    main()


