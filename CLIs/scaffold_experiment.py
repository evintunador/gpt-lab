import argparse
import subprocess
from pathlib import Path


TEMPLATE_YAML = """include_experiments: []
include_packs: []
"""


def scaffold_experiment(repo_root: Path, name: str):
    """Scaffolds a new experiment."""
    exp_dir = repo_root / "experiments" / name
    gl_dir = exp_dir / "gpt_lab"
    artifacts_dir = gl_dir / "artifacts"

    gl_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    (artifacts_dir / ".gitkeep").write_text("")

    yaml_path = exp_dir / "gpt_lab.yaml"
    if not yaml_path.exists():
        yaml_path.write_text(TEMPLATE_YAML)

    main_path = exp_dir / "main.py"
    if not main_path.exists():
        main_path.write_text(
            f"""if __name__ == '__main__':
    print('Hello from experiment: {name}')
"""
        )

    if not (exp_dir / ".git").exists():
        print("Initializing git repository...")
        subprocess.run(["git", "init"], cwd=exp_dir, check=True, capture_output=True)
        subprocess.run(["git", "add", "."], cwd=exp_dir, check=True, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "Initial commit from scaffold"],
            cwd=exp_dir,
            check=True,
            capture_output=True,
        )

    print(f"Scaffolded experiment at {exp_dir}")


def main():
    parser = argparse.ArgumentParser(description="Scaffold a new experiment with gpt_lab layout")
    parser.add_argument("name", help="Experiment name (folder under experiments/)")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    scaffold_experiment(repo, args.name)


if __name__ == "__main__":
    main()


