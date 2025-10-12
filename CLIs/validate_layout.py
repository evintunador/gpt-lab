from pathlib import Path


def main():
    repo = Path(__file__).resolve().parents[2]
    problems = []

    # Ensure markers/roots exist
    if not (repo / ".gpt_lab_root").exists():
        problems.append("Missing .gpt_lab_root marker at repo root")

    # Ensure artifacts folders exist
    expected = [
        repo / "catalogs" / "core" / "gpt_lab" / "artifacts",
        repo / "catalogs" / "packs" / "nlp" / "gpt_lab" / "artifacts",
        repo / "catalogs" / "packs" / "cv" / "gpt_lab" / "artifacts",
    ]
    for p in expected:
        if not p.exists():
            problems.append(f"Missing artifacts directory: {p}")

    if problems:
        for p in problems:
            print("[ERROR]", p)
        raise SystemExit(1)
    print("Layout OK")


if __name__ == "__main__":
    main()


