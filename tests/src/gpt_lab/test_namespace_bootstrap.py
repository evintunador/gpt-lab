import os
import sys
from pathlib import Path
import importlib


def _repo_root() -> Path:
    cur = Path(__file__).resolve()
    for parent in [cur] + list(cur.parents):
        if (parent / ".gpt_lab_root").exists():
            return parent
    # fallback: assume repo root is 3 levels up from src/gpt_lab/tests
    return Path(__file__).resolve().parents[3]


def _reload_gpt_lab():
    # Remove cached modules to force bootstrap to run with new env
    to_delete = [m for m in list(sys.modules.keys()) if m == "gpt_lab" or m.startswith("gpt_lab.")]
    for m in to_delete:
        try:
            del sys.modules[m]
        except Exception:
            pass
    importlib.invalidate_caches()
    return importlib.import_module("gpt_lab")


def _with_env(env_overrides):
    class _EnvCtx:
        def __enter__(self):
            self._old = {}
            for k, v in env_overrides.items():
                self._old[k] = os.environ.get(k)
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
            return self

        def __exit__(self, exc_type, exc, tb):
            for k, old in self._old.items():
                if old is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = old
    return _EnvCtx()


def _expect_ordered_roots(expected_roots):
    # Import after env changes
    _reload_gpt_lab()
    from gpt_lab.catalog_bootstrap import get_active_context
    ctx = get_active_context()
    roots = [Path(p) for p in ctx.get("ordered_roots", [])]
    assert len(roots) >= len(expected_roots), f"ordered_roots too short: {roots}"
    # Ensure expected roots appear in order (allow extras in between)
    iter_roots = iter(roots)
    for exp in expected_roots:
        for r in iter_roots:
            if r == exp:
                break
        else:
            assert False, f"Expected root {exp} not found in ordered_roots: {roots}"
    # Verify package __path__ aligns for one catalog (nn_modules)
    import gpt_lab.nn_modules as nnpkg
    nn_paths = list(getattr(nnpkg, "__path__", []))
    # Verify package __path__ aligns for those roots that have nn_modules
    filtered_expected = [exp for exp in expected_roots if (exp / "nn_modules").is_dir()]
    last_index = -1
    for exp in filtered_expected:
        exp_nn = str(exp / "nn_modules")
        assert exp_nn in nn_paths, f"nn_modules.__path__ missing {exp_nn}"
        idx = nn_paths.index(exp_nn)
        assert idx >= last_index, "nn_modules __path__ order is not non-decreasing"
        last_index = idx


def test_precedence_order_env():
    repo = _repo_root()
    exp_root = repo / "experiments" / "nano_gpt" / "gpt_lab"
    pack_root = repo / "catalogs" / "packs" / "nlp" / "gpt_lab"
    core_root = repo / "catalogs" / "core" / "gpt_lab"
    assert exp_root.is_dir() and pack_root.is_dir() and core_root.is_dir()

    with _with_env({
        "GPT_LAB_CURRENT_EXPERIMENT": "nano_gpt",
        "GPT_LAB_ACTIVE_EXPERIMENTS": "nano_gpt",
        "GPT_LAB_ACTIVE_PACKS": "nlp",
    }):
        _expect_ordered_roots([exp_root, pack_root, core_root])


def test_inactive_roots_ignored():
    repo = _repo_root()
    modded_exp_root = repo / "experiments" / "modded_nano_gpt" / "gpt_lab"
    core_root = repo / "catalogs" / "core" / "gpt_lab"
    assert modded_exp_root.is_dir() and core_root.is_dir()

    with _with_env({
        "GPT_LAB_CURRENT_EXPERIMENT": "modded_nano_gpt",
        "GPT_LAB_ACTIVE_EXPERIMENTS": "modded_nano_gpt",
        "GPT_LAB_ACTIVE_PACKS": "",
    }):
        _expect_ordered_roots([modded_exp_root, core_root])


