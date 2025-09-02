# Inspired by discussions similar to
# https://medium.com/redsquirrel-tech/llm-as-compiler-2a2f79d30f0b

import hashlib
import importlib.util
import io
import json
import os
import sys
import textwrap
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from llm_compiler.llm_interface import LLMClient, DummyLLM, create_llm
from utils import best_device


@dataclass
class CompileConfig:
    out_dir: str = "train_loops/catalog/llm_compiled"
    max_refine_attempts: int = 3
    max_restarts: int = 3
    allow_unreviewed_use: bool = False  # if False, prompt y/n on using unreviewed cached loops
    device: str = best_device


SYSTEM_PROMPT = \
"""You generate Python training loops for PyTorch.
Constraints:
- Output ONLY valid Python code for a single file. No backticks, no prose.
- Provide a function with EXACT signature:
  def run_training(model, optimizer, loss_fn, train_loader, **kwargs) -> dict:
    - Train IN-PLACE on train_loader.
    - Return a dict with at least keys: {'model': nn.Module}.
- Avoid introducing new external dependencies; those used by example scripts are allowed.
- Keep code deterministic where feasible (set seeds when creating schedulers, etc.).
- Do not rely on global variables; everything must be self-contained in this file. Those used by example scripts are exceptions.
- Assume caller moves/creates model/optimizer/loss/data; you just train.
- Err on the side of setting default arguments when reasonable; kwargs should have defaults.

Notes:
- You may add helper functions/classes if needed.
"""

USER_PROMPT_TEMPLATE = \
"""Combine the following atomic features into a single training loop:
{atomic_features}
"""


def _slugify(text: str) -> str:
    safe = []
    for ch in text.lower():
        if ch.isalnum() or ch in "-_":
            safe.append(ch)
        elif ch in " .,/\\:+|[](){}":
            safe.append("-")
    slug = "".join(safe).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug or "loop"


def _make_descriptive_name(atomic_features: List[str]) -> str:
    # Remove .py extension if present
    clean_features = [f.replace('.py', '') for f in atomic_features]
    feature_str = "-".join(sorted(clean_features))
    return f"{_slugify(feature_str)}"


def _write_file(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def _import_module_from_path(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _filter_traceback_for_paths(tb: traceback.TracebackException, focus_paths: List[str]) -> str:
    focus_paths = [str(Path(p)) for p in focus_paths]
    filtered = []
    for frame in tb.stack:
        fn = str(Path(frame.filename))
        if any(fn.endswith(fp) for fp in focus_paths) or "run_training" in frame.name:
            filtered.append(f'  File "{fn}", line {frame.lineno}, in {frame.name}\n    {frame.line or ""}')
    return "\n".join(filtered)


def _summarize_exception_filtered(focus_paths: List[str], phase: str) -> str:
    exc_type, exc, tb = sys.exc_info()
    if exc_type is None:
        return "Unknown error with no traceback."
    tbe = traceback.TracebackException(exc_type, exc, tb, limit=20)
    header = f"{phase} {exc_type.__name__}: {str(exc)}"
    focused = _filter_traceback_for_paths(tbe, focus_paths)
    if not focused:
        # fallback to last few lines of full traceback
        return header + "\n" + "".join(tbe.format())[-2000:]
    return header + "\n" + focused


def _universal_learning_test(run_training_fn, device: str = best_device) -> Dict[str, Any]:
    """
    Build a tiny task and ensure real learning happened (loss drops).
    """
    torch.manual_seed(0)
    X = torch.randn(2048, 32).to(device)
    y = (X.sum(dim=1) > 0).long().to(device)
    ds = TensorDataset(X, y)
    dl = DataLoader(ds, batch_size=64, shuffle=True)

    model = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 2))
    loss_fn = nn.CrossEntropyLoss()
    optim = torch.optim.AdamW(model.parameters(), lr=3e-3)

    model.to(device)
    # Measure pre-training loss
    with torch.no_grad():
        pre = loss_fn(model(X.to(device)), y.to(device)).item()

    result = run_training_fn(
        model=model,
        optimizer=optim,
        loss_fn=loss_fn,
        train_loader=dl,
    )

    # Measure post-training loss
    with torch.no_grad():
        post = loss_fn(model(X.to(device)), y.to(device)).item()

    if not isinstance(result, dict):
        raise AssertionError("run_training(...) must return dict metrics.")
    if not (post < pre * 0.9):  # at least 10% relative improvement
        raise AssertionError(f"Training did not sufficiently improve loss: pre={pre:.4f}, post={post:.4f}")

    return {"pre_loss": pre, "post_loss": post, **result}


def _atomic_dir() -> Path:
    return Path(__file__).resolve().parent / "catalog" / "atomic_features"


def _get_atomic_files(atomic_features: List[str]) -> List[Path]:
    """
    Get paths to the specified atomic feature files.
    """
    paths: List[Path] = []
    root = _atomic_dir()

    for feature in atomic_features:
        # Add .py extension if not present
        filename = feature if feature.endswith('.py') else f"{feature}.py"
        p = root / filename
        if p.exists() and p.is_file() and p.stat().st_size > 0:
            paths.append(p)
        else:
            # List available features for better error messages
            available = [f.stem for f in root.glob("*.py") if f.is_file()]
            raise ValueError(f"Atomic feature file not found: {feature}\nAvailable features: {', '.join(available)}")

    return paths


def _read_atomic_examples_text(paths: List[Path], char_budget: int = 8000) -> str:
    """
    Read selected atomic files and assemble a prompt appendix with fenced examples.
    Truncates if needed to stay within a rough character budget.
    """
    chunks: List[str] = []
    used = 0
    for p in paths:
        try:
            code = p.read_text(encoding="utf-8")
        except Exception:
            continue
        header = f"# Example from {p.name}\n"
        block = f"```python\n{header}{code}\n```\n"
        if used + len(block) > char_budget and chunks:
            break
        chunks.append(block)
        used += len(block)
    return "\n".join(chunks)


def _build_user_prompt(atomic_features: List[str]) -> str:
    atomic_features_str = ", ".join(atomic_features)
    base = USER_PROMPT_TEMPLATE.format(atomic_features=atomic_features_str)
    
    try:
        paths = _get_atomic_files(atomic_features)
        if paths:
            examples = _read_atomic_examples_text(paths)
            if examples:
                base = base + "\nAtomic feature examples to combine:\n" + examples
    except ValueError as e:
        # If we can't find the files, still continue but mention it
        base = base + f"\nNote: {str(e)}"
    
    return base


def compile_loop(
    atomic_features: List[str],
    llm: Optional[LLMClient] = None,
    cfg: CompileConfig = CompileConfig(),
) -> Dict[str, Any]:
    """
    Main entry: ask LLM for a bespoke training loop combining atomic features, test it, cache it.
    Returns manifest dict with paths and metadata.
    """
    llm = llm or DummyLLM()
    name = _make_descriptive_name(atomic_features)
    root = Path(cfg.out_dir) / name
    code_path = root / f"{name}.py"
    manifest_path = root / "manifest.json"
    review_flag_path = root / "human_reviewed.json"

    # Cached success path
    if manifest_path.exists() and code_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        reviewed = False
        if review_flag_path.exists():
            with open(review_flag_path, "r", encoding="utf-8") as f:
                try:
                    reviewed = json.load(f).get("reviewed", False)
                except Exception:
                    reviewed = False
        if not reviewed and not cfg.allow_unreviewed_use:
            ans = input(f"Cached loop '{name}' is not human-reviewed. Proceed? [y/N]: ").strip().lower()
            if ans not in ("y", "yes", "Yes"):
                print("Aborting by user choice.")
                sys.exit(1)
        print(f"[cache] Using cached compiled loop at {code_path}")
        return {"name": name, "code_path": str(code_path), **manifest}

    # Build prompts
    user_prompt = _build_user_prompt(atomic_features)

    # Generate + refine loop
    restarts_left = cfg.max_restarts
    last_error_summary = ""
    code = ""
    while restarts_left >= 0:
        try:
            code = llm.generate(SYSTEM_PROMPT, user_prompt) if not last_error_summary \
                else llm.refine(SYSTEM_PROMPT, user_prompt, prior_code=code, error_summary=last_error_summary)

            # Write and import
            _write_file(code_path, code)
            try:
                module = _import_module_from_path(f"compiled_loop", code_path)
            except Exception:
                err = _summarize_exception_filtered([str(code_path)], phase="[import]")
                raise RuntimeError(err)

            if not hasattr(module, "run_training"):
                raise AssertionError("Generated file must define function 'run_training' with the required signature.")
            run_training_fn = getattr(module, "run_training")

            # Test
            try:
                metrics = _universal_learning_test(run_training_fn, device=cfg.device)
            except Exception:
                err = _summarize_exception_filtered([str(code_path)], phase="[test]")
                raise RuntimeError(err)

            # Filter out non-serializable objects from metrics before saving
            serializable_metrics = {k: v for k, v in metrics.items() if k != "model"}
            manifest = {
                "name": name,
                "atomic_features": atomic_features,
                "metrics": serializable_metrics,
                "device": cfg.device,
            }
            _write_file(manifest_path, json.dumps(manifest, indent=2))
            _write_file(review_flag_path, json.dumps({"reviewed": False, "reviewer": None, "notes": ""}, indent=2))

            print(f"[ok] Compiled and validated. Cached at {root}")
            return {"name": name, "code_path": str(code_path), **manifest}

        except Exception as e:
            # Pass only focused, phase-tagged errors into refine loop
            err = str(e)
            print(f"[compile/test error]\n{err}")
            # Try refine attempts first
            for _ in range(cfg.max_refine_attempts):
                try:
                    last_error_summary = err
                    code = llm.refine(SYSTEM_PROMPT, user_prompt, prior_code=code, error_summary=err)
                    _write_file(code_path, code)
                    try:
                        module = _import_module_from_path(f"compiled_loop", code_path)
                    except Exception:
                        err = _summarize_exception_filtered([str(code_path)], phase="[import]")
                        raise RuntimeError(err)
                    if not hasattr(module, "run_training"):
                        raise AssertionError("Generated file must define function 'run_training'.")
                    run_training_fn = getattr(module, "run_training")
                    try:
                        metrics = _universal_learning_test(run_training_fn, device=cfg.device)
                    except Exception:
                        err = _summarize_exception_filtered([str(code_path)], phase="[test]")
                        raise RuntimeError(err)
                    # Filter out non-serializable objects from metrics before saving
                    serializable_metrics = {k: v for k, v in metrics.items() if k != "model"}
                    manifest = {
                        "name": name,
                        "atomic_features": atomic_features,
                        "metrics": serializable_metrics,
                        "device": cfg.device,
                    }
                    _write_file(manifest_path, json.dumps(manifest, indent=2))
                    _write_file(review_flag_path, json.dumps({"reviewed": False, "reviewer": None, "notes": ""}, indent=2))
                    print(f"[ok after refine] Cached at {root}")
                    return {"name": name, "code_path": str(code_path), **manifest}
                except Exception as e2:
                    err = str(e2)
                    print(f"[refine error]\n{err}")
                    continue
            # Restart from scratch
            restarts_left -= 1
            if restarts_left < 0:
                raise
            print("[restart] Starting a fresh attempt...")
            last_error_summary = ""


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("atomic_features", nargs='+', help="List of atomic feature filenames (e.g., grad_accum.py grad_norm_clip.py)")
    parser.add_argument("--device", type=str, default=best_device)
    parser.add_argument("--allow_unreviewed_use", action="store_true")
    parser.add_argument("--model", type=str, default="openai/gpt-4o-mini", help="Provider/model string, e.g., 'openai/gpt-4o', 'anthropic/claude-3-5-sonnet-20240620.")
    parser.add_argument("--api_key", type=str, default=None, help="Optional API key; otherwise use env vars.")
    args = parser.parse_args()

    cfg = CompileConfig(device=args.device, allow_unreviewed_use=args.allow_unreviewed_use)

    llm = create_llm(args.model, api_key=args.api_key)
    manifest = compile_loop(args.atomic_features, llm=llm, cfg=cfg)
    print(json.dumps(manifest, indent=2))