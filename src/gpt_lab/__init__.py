# Initialize namespace paths: default to core-only; auto-layer in experiments or when requested.
try:
    import os
    from pathlib import Path
    from .catalog_bootstrap import bootstrap_namespace_paths, get_repo_root

    should_layer = bool(
        os.getenv("GPT_LAB_BOOTSTRAP", "")
        or os.getenv("GPT_LAB_CURRENT_EXPERIMENT", "")
        or os.getenv("GPT_LAB_ACTIVE_EXPERIMENTS", "")
        or os.getenv("GPT_LAB_ACTIVE_PACKS", "")
    )

    if not should_layer:
        try:
            repo_root = get_repo_root()
            cwd = Path.cwd().resolve()
            rel = cwd.relative_to(repo_root)
            parts = rel.parts
            in_experiment = len(parts) >= 2 and parts[0] == "experiments"
            should_layer = in_experiment
        except Exception:
            pass

    if should_layer:
        os.environ.pop("GPT_LAB_CORE_ONLY", None)
        bootstrap_namespace_paths()
    else:
        os.environ.setdefault("GPT_LAB_CORE_ONLY", "1")
        bootstrap_namespace_paths()
except Exception:
    # Avoid hard failures on import; discovery tools may still work partially.
    pass
