from pathlib import Path


def validate_layout(repo_root: Path) -> list[str]:
    """Validates the gpt_lab repository layout.
    
    Args:
        repo_root: Path to the repository root
        
    Returns:
        List of problem descriptions (empty if layout is valid)
    """
    problems = []

    # Ensure markers/roots exist
    if not (repo_root / ".gpt_lab_root").exists():
        problems.append("Missing .gpt_lab_root marker at repo root")

    # Ensure artifacts folders exist
    expected = [
        repo_root / "catalogs" / "core" / "gpt_lab" / "artifacts",
        repo_root / "catalogs" / "packs" / "nlp" / "gpt_lab" / "artifacts",
        repo_root / "catalogs" / "packs" / "cv" / "gpt_lab" / "artifacts",
    ]
    for p in expected:
        if not p.exists():
            problems.append(f"Missing artifacts directory: {p}")

    return problems


def main():
    repo = Path(__file__).resolve().parents[1]
    problems = validate_layout(repo)

    if problems:
        for p in problems:
            print("[ERROR]", p)
        raise SystemExit(1)
    print("Layout OK")


if __name__ == "__main__":
    main()


