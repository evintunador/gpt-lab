import argparse
import importlib
from typing import Optional

from gpt_lab.catalog_bootstrap import get_active_context


def format_context_output(ctx: dict, verbose: bool = False) -> str:
    """Formats the active context information into a string.
    
    Args:
        ctx: Context dictionary from get_active_context()
        verbose: Whether to include verbose package path information
        
    Returns:
        Formatted string output
    """
    lines = []
    lines.append(f"repo_root: {ctx.get('repo_root')}")
    lines.append(f"current_experiment: {ctx.get('current_experiment')}")
    lines.append(f"active_experiments: {ctx.get('active_experiments')}")
    lines.append(f"active_packs: {ctx.get('active_packs')}")
    lines.append("ordered_roots:")
    for r in ctx.get("ordered_roots", []):
        lines.append(f" - {r}")

    if verbose:
        lines.append("")
        lines.append("package __path__s:")
        cats = ["nn_modules", "optimizers", "train_loops", "benchmarks", "data_sources", "models", "llm_code_compilers"]
        for cat in cats:
            try:
                pkg = importlib.import_module(f"gpt_lab.{cat}")
                lines.append(f"gpt_lab.{cat}:")
                for p in getattr(pkg, "__path__", []):
                    lines.append(f"  - {p}")
            except Exception as e:
                lines.append(f"gpt_lab.{cat}: <unavailable> ({e})")

    return "\n".join(lines)


def print_active_paths(verbose: bool = False, context: Optional[dict] = None):
    """Prints the active gpt_lab paths and context.
    
    Args:
        verbose: Whether to include verbose package path information
        context: Optional pre-fetched context (for testing). If None, will call get_active_context()
    """
    if context is None:
        context = get_active_context()
    
    output = format_context_output(context, verbose=verbose)
    print(output)


def main():
    parser = argparse.ArgumentParser(description="Print active gpt_lab roots and package __path__s")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print full context")
    args = parser.parse_args()

    print_active_paths(verbose=args.verbose)


if __name__ == "__main__":
    main()
