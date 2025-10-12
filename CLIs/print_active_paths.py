import argparse
from pprint import pprint

from gpt_lab.catalog_bootstrap import get_active_context


def main():
    parser = argparse.ArgumentParser(description="Print active gpt_lab roots and package __path__s")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print full context")
    args = parser.parse_args()

    ctx = get_active_context()
    print("repo_root:", ctx.get("repo_root"))
    print("current_experiment:", ctx.get("current_experiment"))
    print("active_experiments:", ctx.get("active_experiments"))
    print("active_packs:", ctx.get("active_packs"))
    print("ordered_roots:")
    for r in ctx.get("ordered_roots", []):
        print(" -", r)

    if args.verbose:
        try:
            import importlib
            cats = ["nn_modules", "optimizers", "train_loops", "benchmarks", "data_sources", "models", "llm_code_compilers"]
            print("\npackage __path__s:")
            for cat in cats:
                try:
                    pkg = importlib.import_module(f"gpt_lab.{cat}")
                    print(f"gpt_lab.{cat}:")
                    for p in getattr(pkg, "__path__", []):
                        print("  -", p)
                except Exception as e:
                    print(f"gpt_lab.{cat}: <unavailable> ({e})")
        except Exception:
            pass


if __name__ == "__main__":
    main()


