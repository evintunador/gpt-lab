import os
import sys
import logging
from pathlib import Path
from typing import List, Dict, Tuple

import importlib


logger = logging.getLogger(__name__)


# Catalog types we support as unified namespaces.
# It's OK if some types are empty in some roots.
CATALOG_TYPES: List[str] = [
    "nn_modules",
    "optimizers",
    "train_loops",
    "benchmarks",
    "data_sources",
    "models",
    "llm_code_compilers",
]


def _is_repo_root(path: Path) -> bool:
    # Prefer explicit sentinel file to avoid misclassifying experiment dirs
    if (path / ".gpt_lab_root").exists():
        return True
    # Heuristic: top-level repo should have both src/ and catalogs/
    if (path / "src").is_dir() and (path / "catalogs").is_dir():
        return True
    return False


def _find_repo_root() -> Path:
    # 1) Explicit override
    env_root = os.getenv("GPT_LAB_ROOT")
    if env_root:
        p = Path(env_root).resolve()
        if p.exists():
            return p

    # 2) Walk up from CWD
    cur = Path.cwd().resolve()
    for parent in [cur] + list(cur.parents):
        if _is_repo_root(parent):
            return parent

    # 3) Walk up from this file
    here = Path(__file__).resolve()
    for parent in [here] + list(here.parents):
        if _is_repo_root(parent):
            return parent

    # 4) Fallback: project root assumed two up from src/gpt_lab
    return Path(__file__).resolve().parents[2]


def _load_repo_yaml(repo_root: Path) -> Dict:
    try:
        import yaml  # optional; only used if present
    except Exception:
        return {}

    cfg_path = repo_root / "gpt_lab.yaml"
    if not cfg_path.exists():
        return {}
    try:
        return yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def _infer_current_experiment(repo_root: Path) -> str:
    # In core-only mode, do not infer any experiment
    if os.getenv("GPT_LAB_CORE_ONLY", ""):
        return ""
    # 1) Explicit env
    cur_exp = os.getenv("GPT_LAB_CURRENT_EXPERIMENT")
    if cur_exp:
        return cur_exp

    # 2) First active experiments list
    active = os.getenv("GPT_LAB_ACTIVE_EXPERIMENTS", "").strip()
    if active:
        return active.split(",")[0].strip()

    # 3) Infer from CWD if inside experiments/<name>/
    cwd = Path.cwd().resolve()
    try:
        rel = cwd.relative_to(repo_root)
        parts = rel.parts
        if len(parts) >= 2 and parts[0] == "experiments":
            return parts[1]
    except Exception:
        pass

    return ""


def _load_experiment_yaml(repo_root: Path, exp_name: str) -> Dict:
    if not exp_name:
        return {}
    try:
        import yaml
    except Exception:
        return {}
    cfg_path = repo_root / "experiments" / exp_name / "gpt_lab.yaml"
    if not cfg_path.exists():
        return {}
    try:
        return yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def _parse_csv_env(name: str) -> List[str]:
    raw = os.getenv(name, "").strip()
    if not raw:
        return []
    return [s.strip() for s in raw.split(",") if s.strip()]


def _resolve_active_roots(repo_root: Path) -> Tuple[str, List[str], List[str]]:
    # Core-only mode disables experiments and packs
    if os.getenv("GPT_LAB_CORE_ONLY", ""):
        return "", [], []
    repo_cfg = _load_repo_yaml(repo_root)
    cur_exp = _infer_current_experiment(repo_root)
    exp_cfg = _load_experiment_yaml(repo_root, cur_exp)

    # Sources in increasing priority: repo YAML < experiment YAML < ENV
    yaml_exps = list(exp_cfg.get("include_experiments", []) or [])
    if not yaml_exps:
        yaml_exps = list((repo_cfg.get("include_experiments", []) or []))
    env_exps = _parse_csv_env("GPT_LAB_ACTIVE_EXPERIMENTS")
    active_experiments = env_exps or yaml_exps

    yaml_packs = list(exp_cfg.get("include_packs", []) or [])
    if not yaml_packs:
        yaml_packs = list((repo_cfg.get("include_packs", []) or []))
    env_packs = _parse_csv_env("GPT_LAB_ACTIVE_PACKS")
    active_packs = env_packs or yaml_packs

    return cur_exp, active_experiments, active_packs


def _ordered_search_roots(repo_root: Path, cur_exp: str, active_experiments: List[str], active_packs: List[str]) -> List[Path]:
    roots: List[Path] = []

    # Experiments: current first, then the rest in declared order (excluding current), then tie-break alphabetically if none provided
    exp_root = repo_root / "experiments"
    seen = set()
    if cur_exp:
        p = exp_root / cur_exp / "gpt_lab"
        if p.is_dir():
            roots.append(p)
            seen.add(cur_exp)

    for name in active_experiments:
        if name and name not in seen:
            p = exp_root / name / "gpt_lab"
            if p.is_dir():
                roots.append(p)
                seen.add(name)

    # Packs in declared order
    packs_root = repo_root / "catalogs" / "packs"
    for name in active_packs:
        p = packs_root / name / "gpt_lab"
        if p.is_dir():
            roots.append(p)

    # Core (always last)
    core = repo_root / "catalogs" / "core" / "gpt_lab"
    if core.is_dir():
        roots.append(core)

    return roots


def _ensure_package_importable(package_name: str) -> bool:
    try:
        importlib.import_module(package_name)
        return True
    except Exception:
        return False


def bootstrap_namespace_paths() -> None:
    """
    Compute active roots (experiments, packs, core) and extend namespace package
    __path__ for each catalog type accordingly, in precedence order.
    """
    try:
        repo_root = _find_repo_root()
    except Exception:
        # If this fails, do nothing rather than crash import-time
        return

    cur_exp, active_exps, active_packs = _resolve_active_roots(repo_root)
    roots = _ordered_search_roots(repo_root, cur_exp, active_exps, active_packs)

    # Optional debug output
    if os.getenv("GPT_LAB_LOGLEVEL", "").lower() in ("debug", "info"):
        logger.info("gpt_lab bootstrap: repo_root=%s", repo_root)
        logger.info("gpt_lab bootstrap: current_experiment=%s", cur_exp)
        logger.info("gpt_lab bootstrap: active_experiments=%s", active_exps)
        logger.info("gpt_lab bootstrap: active_packs=%s", active_packs)
        logger.info("gpt_lab bootstrap: roots (in order)=%s", roots)

    # Extend package __path__ for each catalog type
    for cat in CATALOG_TYPES:
        pkg_name = f"gpt_lab.{cat}"
        # Ensure the base package exists; if not, skip
        if not _ensure_package_importable(pkg_name):
            continue
        try:
            pkg = importlib.import_module(pkg_name)
        except Exception:
            continue
        # We add the base type dir (not deeper) so subpackages resolve naturally
        # Iterate in reverse and insert at front so the first root ends up first
        for base in reversed(roots):
            candidate = base / cat
            if candidate.is_dir():
                # Avoid duplicates
                candidate_str = str(candidate)
                path_list = getattr(pkg, "__path__", [])
                if candidate_str not in path_list:
                    try:
                        path_list.insert(0, candidate_str)  # type: ignore[attr-defined]
                    except Exception:
                        # Some packages may not expose __path__ properly; skip
                        pass


def get_repo_root() -> Path:
    """Public helper to obtain the resolved repository root."""
    return _find_repo_root()


def get_active_context() -> Dict[str, object]:
    """
    Return a dictionary with active catalog context:
    - repo_root: Path
    - current_experiment: str
    - active_experiments: List[str]
    - active_packs: List[str]
    - ordered_roots: List[Path] (each is a gpt_lab root directory)
    - current_experiment_root: Path or None (gpt_lab dir for current experiment)
    """
    repo_root = _find_repo_root()
    cur_exp, active_exps, active_packs = _resolve_active_roots(repo_root)
    ordered_roots = _ordered_search_roots(repo_root, cur_exp, active_exps, active_packs)

    cur_exp_root = None
    if cur_exp:
        p = repo_root / "experiments" / cur_exp / "gpt_lab"
        if p.is_dir():
            cur_exp_root = p

    return {
        "repo_root": repo_root,
        "current_experiment": cur_exp,
        "active_experiments": active_exps,
        "active_packs": active_packs,
        "ordered_roots": ordered_roots,
        "current_experiment_root": cur_exp_root,
    }


def get_current_experiment_root() -> Path:
    """Return the current experiment's gpt_lab root path if available, else empty Path()."""
    ctx = get_active_context()
    cur = ctx.get("current_experiment_root")
    return cur if isinstance(cur, Path) else Path()


def get_artifact_root() -> Path:
    """
    Resolve the artifact root directory following precedence:
    1) GPT_LAB_ARTIFACT_ROOT env override
    2) Current experiment's gpt_lab/artifacts/
    3) Core catalogs/gpt_lab/artifacts/
    Ensures the directory exists.
    """
    env_path = os.getenv("GPT_LAB_ARTIFACT_ROOT")
    if env_path:
        p = Path(env_path).resolve()
        p.mkdir(parents=True, exist_ok=True)
        return p

    ctx = get_active_context()
    cur = ctx.get("current_experiment_root")
    if isinstance(cur, Path) and cur:
        p = cur / "artifacts"
        p.mkdir(parents=True, exist_ok=True)
        return p

    # Fallback to core
    repo_root = ctx.get("repo_root") or _find_repo_root()
    core = Path(repo_root) / "catalogs" / "core" / "gpt_lab" / "artifacts"
    core.mkdir(parents=True, exist_ok=True)
    return core


def get_all_artifact_roots_for_active() -> List[Path]:
    """
    Return artifact roots for each active ordered root (experiments, packs, core),
    creating them if missing.
    """
    ctx = get_active_context()
    roots = []
    for base in ctx.get("ordered_roots", []):
        p = Path(base) / "artifacts"
        if not p.exists():
            try:
                p.mkdir(parents=True, exist_ok=True)
            except Exception:
                continue
        roots.append(p)
    return roots


