# Inspired by discussions similar to
# https://medium.com/redsquirrel-tech/llm-as-compiler-2a2f79d30f0b

import sys
import traceback
import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable

from train_loops.catalog_test import universal_learning_test, discover_specific_tests
from utils.llm_code_compiler import LLMClient, create_llm
from utils.device import best_device as device
from utils.testing import import_module_from_path

# Global verbose flag
VERBOSE = False

def vprint(*args, **kwargs):
    """Verbose print - only prints if VERBOSE is True"""
    if VERBOSE:
        print(*args, **kwargs)

def vprint_section(title: str):
    """Print a section header for verbose output"""
    if VERBOSE:
        print(f"\n{'='*60}")
        print(f" {title}")
        print(f"{'='*60}")

def vprint_subsection(title: str):
    """Print a subsection header for verbose output"""
    if VERBOSE:
        print(f"\n{'-'*40}")
        print(f" {title}")
        print(f"{'-'*40}")


SYSTEM_PROMPT = \
"""You generate Python training loops for PyTorch.
Constraints:
- Output ONLY valid Python code for a single file. No backticks, no prose.
- Provide a function with EXACT signature:
  def run_training(model, optimizer, loss_fn, train_loader, **kwargs) -> dict:
    - Train IN-PLACE on train_loader.
    - Return a dict with at least keys: {'model': nn.Module}; depending on the atomic feature examples provided there my be others which are relevant.
- Avoid introducing new external dependencies; those used by example scripts are allowed.
- Keep code deterministic where feasible (set seeds when creating schedulers, etc.).
- Do not rely on global variables; everything must be self-contained in this file. Those used by example scripts are exceptions.
- Assume caller moves/creates model/optimizer/loss/data; you just train.
- Err on the side of setting default arguments when reasonable; kwargs should have defaults. 
- All kwarg defaults should be set to values that ensure numerical equivalence with `base_loop.py`.

Testing Requirements:
- Your code will be tested with a universal learning test (loss must decrease by at least 10%)
- Your code will be tested to be numerically equivalent to `base_loop.py` (shown below) when default kwargs values are used.
- Your code will also be tested with specific feature tests described below
- Make sure your implementation correctly handles all the specific behaviors being tested

Notes:
- You may add helper functions/classes if needed, or, if re-using, import directly from one of the atomic features by using `from train_loops.catalog.atomic_features.<feature_name> import <function/class_name>`.
"""

USER_PROMPT_TEMPLATE = \
"""Combine the following atomic features into a single training loop:
{atomic_features}

{test_descriptions}
"""


def _build_system_prompt_with_base_loop() -> str:
    """Build the system prompt including base_loop.py content for reference."""
    base_loop_path = Path("src/train_loops/catalog/atomic_features/base_loop.py")
    
    try:
        base_loop_content = base_loop_path.read_text(encoding="utf-8")
        base_loop_section = f"""

Base Loop Reference (base_loop.py):
Your generated code must be numerically equivalent to this when default kwargs are used:

```python
{base_loop_content}
```
"""
    except Exception:
        base_loop_section = "\n(Note: Could not load base_loop.py for reference)"
    
    return SYSTEM_PROMPT + base_loop_section


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


def _parse_filename_to_atomic_features(filename: str) -> List[str]:
    """Extract atomic features from a compiled loop filename."""
    # Remove .py extension and split on hyphens
    name = filename.replace('.py', '')
    return name.split('-')


def _write_file(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


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


def _read_atomic_examples_text(paths: List[Path], char_budget: int = 100_000) -> str:
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


def _extract_test_descriptions(atomic_features: List[str]) -> str:
    """Extract test descriptions from specific test functions for the given atomic features."""
    specific_tests = discover_specific_tests()
    test_descriptions = []
    
    for feature in atomic_features:
        clean_feature = feature.replace('.py', '')
        if clean_feature in specific_tests:
            test_descriptions.append(f"\n=== Tests for {feature} ===")
            for test_func in specific_tests[clean_feature]:
                # Get function name and docstring
                func_name = test_func.__name__
                docstring = inspect.getdoc(test_func) or "No description available"
                
                # Try to get a simplified version of the test logic
                try:
                    source_lines = inspect.getsourcelines(test_func)[0]
                    # Extract key assertions and logic (simplified)
                    key_lines = []
                    for line in source_lines:
                        line = line.strip()
                        if (line.startswith('assert ') or 
                            'torch.allclose' in line or
                            'accum_steps' in line or
                            'batch_size' in line or
                            'scheduler' in line or
                            'clip_grad' in line or
                            line.startswith('# Test') or
                            line.startswith('"""')):
                            key_lines.append(f"    {line}")
                    
                    if key_lines:
                        test_descriptions.append(f"""
Test: {func_name}
Description: {docstring}
Key test logic:
{''.join(key_lines[:10])}  # ... (truncated for brevity)
""")
                    else:
                        test_descriptions.append(f"""
Test: {func_name}
Description: {docstring}
""")
                except:
                    # Fallback if source inspection fails
                    test_descriptions.append(f"""
Test: {func_name}
Description: {docstring}
""")
    
    if test_descriptions:
        header = """
IMPORTANT: Your generated code will be tested with the following specific tests.
Make sure your implementation satisfies these requirements:
"""
        return header + "\n".join(test_descriptions) + "\n"
    else:
        return ""


def _build_user_prompt(atomic_features: List[str]) -> str:
    atomic_features_str = ", ".join(atomic_features)
    
    # Get test descriptions
    test_descriptions = _extract_test_descriptions(atomic_features)
    
    base = USER_PROMPT_TEMPLATE.format(
        atomic_features=atomic_features_str,
        test_descriptions=test_descriptions
    )
    
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


def _add_metadata_to_code(code: str, atomic_features: List[str], device: str) -> str:
    """Add metadata as comments and dunder variables to the generated code."""
    # Clean the feature names to remove .py extension for consistent storage
    clean_features = [f.replace('.py', '') for f in atomic_features]
    metadata_header = f'''"""
LLM-compiled training loop combining atomic features: {', '.join(atomic_features)}
Generated by gpt-lab LLM compiler
Device: {device}
"""

# Metadata for discovery and testing
__atomic_features__ = {clean_features!r}
__llm_compiled__ = True

'''
    return metadata_header + code


def run_specific_tests_for_compilation(run_training_fn: Callable, atomic_features: List[str]):
    """Run all applicable specific tests during compilation - validation only."""
    specific_tests = discover_specific_tests()
    
    for feature in atomic_features:
        clean_feature = feature.replace('.py', '')  # Handle both formats
        if clean_feature in specific_tests:
            for test_func in specific_tests[clean_feature]:
                # Just run the test - if it fails, compilation fails
                test_func(run_training_fn)


def run_base_loop_compliance_test_for_compilation(run_training_fn: Callable, atomic_features: List[str]):
    """Run base_loop compliance test during compilation."""
    from train_loops.catalog_test import base_loop_compliance_test
    
    # For compiled loops, we use a representative name from the atomic features
    feature_name = f"compiled_loop_{'-'.join(sorted([f.replace('.py', '') for f in atomic_features]))}"
    base_loop_compliance_test(run_training_fn, feature_name)


def compile_loop(
    atomic_features: List[str],
    llm: Optional[LLMClient] = None,
    max_refine_attempts: int = 3, 
    max_restarts: int = 3,
) -> Dict[str, Any]:
    """
    Main entry: ask LLM for a bespoke training loop combining atomic features, test it, cache it.
    """
    llm = llm or LLMClient()
    name = _make_descriptive_name(atomic_features)
    code_path = Path("src/train_loops/catalog/llm_compiled") / f"{name}.py"

    vprint_section("LLM TRAINING LOOP COMPILATION")
    vprint(f"Atomic features: {atomic_features}")
    vprint(f"Generated name: {name}")
    vprint(f"Output path: {code_path}")

    # Cached success path
    if code_path.exists():
        vprint_subsection("CACHE CHECK")
        vprint(f"Found existing file at {code_path}")
        try:
            module = import_module_from_path(f"cached_loop", code_path)
            vprint("✓ Successfully loaded cached loop")
            print(f"[cache] Using cached compiled loop at {code_path}")
            return {
                "name": name, 
                "code_path": str(code_path), 
                "atomic_features": atomic_features,
            }
        except Exception as e:
            vprint(f"✗ Failed to load cached loop: {e}")
            print(f"[warning] Failed to load cached loop {code_path}: {e}")
            print("[info] Will regenerate...")

    # Build prompts
    vprint_subsection("PROMPT CONSTRUCTION")
    system_prompt = _build_system_prompt_with_base_loop()
    user_prompt = _build_user_prompt(atomic_features)
    
    vprint("System prompt:")
    vprint(f"```\n{system_prompt}\n```")
    vprint("\nUser prompt:")
    vprint(f"```\n{user_prompt}\n```")

    # Generate + refine loop
    restarts_left = max_restarts
    last_error_summary = ""
    code = ""
    attempt_num = 1
    
    vprint_subsection("CODE GENERATION AND VALIDATION")
    
    while restarts_left >= 0:
        try:
            if not last_error_summary:
                vprint(f"\n🚀 ATTEMPT {attempt_num}: Initial generation")
                vprint("Calling LLM.generate()...")
                code = llm.generate(system_prompt, user_prompt)
            else:
                vprint(f"\n🔄 ATTEMPT {attempt_num}: Refinement")
                vprint("Calling LLM.refine()...")
                vprint(f"Error to fix: {last_error_summary}")
                code = llm.refine(system_prompt, user_prompt, prior_code=code, error_summary=last_error_summary)
            
            vprint("✓ LLM response received")
            vprint(f"Generated code length: {len(code)} characters")
            if VERBOSE:
                print("Generated code preview:")
                print(f"```python\n{code[:5000]}{'...' if len(code) > 5000 else ''}\n```")

            # Add metadata and write
            vprint("\n📝 Adding metadata and writing file...")
            code_with_metadata = _add_metadata_to_code(code, atomic_features, device)
            _write_file(code_path, code_with_metadata)
            vprint(f"✓ File written to {code_path}")
            
            vprint("\n🔍 Testing generated code...")
            
            # Import test
            vprint("Testing import...")
            try:
                module = import_module_from_path(f"compiled_loop", code_path)
                vprint("✓ Import successful")
            except Exception:
                err = _summarize_exception_filtered([str(code_path)], phase="[import]")
                vprint(f"✗ Import failed: {err}")
                raise RuntimeError(err)

            # Function signature test
            vprint("Checking for run_training function...")
            if not hasattr(module, "run_training"):
                vprint("✗ Missing run_training function")
                raise AssertionError("Generated file must define function 'run_training' with the required signature.")
            run_training_fn = getattr(module, "run_training")
            vprint("✓ run_training function found")

            # Universal test
            vprint("Running universal learning test...")
            try:
                universal_learning_test(run_training_fn)
                vprint("✓ Universal test passed")
            except Exception:
                err = _summarize_exception_filtered([str(code_path)], phase="[universal_test]")
                vprint(f"✗ Universal test failed: {err}")
                raise RuntimeError(err)

            # Base loop compliance test
            vprint("Running base_loop compliance test...")
            try:
                run_base_loop_compliance_test_for_compilation(run_training_fn, atomic_features)
                vprint("✓ Base loop compliance test passed")
            except Exception:
                err = _summarize_exception_filtered([str(code_path)], phase="[base_loop_compliance]")
                vprint(f"✗ Base loop compliance test failed: {err}")
                raise RuntimeError(err)

            # Specific tests
            vprint("Running specific feature tests...")
            try:
                run_specific_tests_for_compilation(run_training_fn, atomic_features)
                vprint("✓ All specific tests passed")
            except Exception:
                err = _summarize_exception_filtered([str(code_path)], phase="[specific_tests]")
                vprint(f"✗ Specific tests failed: {err}")
                raise RuntimeError(err)

            vprint_section("COMPILATION SUCCESSFUL")
            print(f"[ok] Compiled and validated. Cached at {code_path}")
            return {
                "name": name, 
                "code_path": str(code_path), 
                "atomic_features": atomic_features,
            }

        except Exception as e:
            # Pass only focused, phase-tagged errors into refine loop
            err = str(e)
            vprint(f"\n❌ ATTEMPT {attempt_num} FAILED: {err}")
            print(f"[compile/test error]\n{err}")
            
            # Try refine attempts first
            vprint(f"\n🔧 Starting {max_refine_attempts} refinement attempts...")
            for refine_attempt in range(max_refine_attempts):
                try:
                    vprint(f"\n🔧 REFINEMENT {refine_attempt + 1}/{max_refine_attempts}")
                    last_error_summary = err
                    vprint("Calling LLM.refine()...")
                    code = llm.refine(system_prompt, user_prompt, prior_code=code, error_summary=err)
                    vprint("✓ Refinement response received")
                    
                    code_with_metadata = _add_metadata_to_code(code, atomic_features, device)
                    _write_file(code_path, code_with_metadata)
                    vprint(f"✓ Refined code written to {code_path}")
                    
                    # Test refined code
                    vprint("Testing refined code...")
                    try:
                        module = import_module_from_path(f"compiled_loop", code_path)
                        vprint("✓ Import successful")
                    except Exception:
                        err = _summarize_exception_filtered([str(code_path)], phase="[import]")
                        vprint(f"✗ Import failed: {err}")
                        raise RuntimeError(err)
                        
                    if not hasattr(module, "run_training"):
                        vprint("✗ Missing run_training function")
                        raise AssertionError("Generated file must define function 'run_training'.")
                    run_training_fn = getattr(module, "run_training")
                    vprint("✓ run_training function found")
                    
                    try:
                        universal_learning_test(run_training_fn)
                        vprint("✓ Universal test passed")
                    except Exception:
                        err = _summarize_exception_filtered([str(code_path)], phase="[universal_test]")
                        vprint(f"✗ Universal test failed: {err}")
                        raise RuntimeError(err)
                    
                    try:
                        run_base_loop_compliance_test_for_compilation(run_training_fn, atomic_features)
                        vprint("✓ Base loop compliance test passed")
                    except Exception:
                        err = _summarize_exception_filtered([str(code_path)], phase="[base_loop_compliance]")
                        vprint(f"✗ Base loop compliance test failed: {err}")
                        raise RuntimeError(err)
                        
                    try:
                        run_specific_tests_for_compilation(run_training_fn, atomic_features)
                        vprint("✓ All specific tests passed")
                    except Exception:
                        err = _summarize_exception_filtered([str(code_path)], phase="[specific_tests]")
                        vprint(f"✗ Specific tests failed: {err}")
                        raise RuntimeError(err)
                        
                    vprint_section("REFINEMENT SUCCESSFUL")
                    print(f"[ok after refine] Cached at {code_path}")
                    return {
                        "name": name, 
                        "code_path": str(code_path), 
                        "atomic_features": atomic_features,
                    }
                except Exception as e2:
                    err = str(e2)
                    vprint(f"✗ REFINEMENT {refine_attempt + 1} FAILED: {err}")
                    print(f"[refine error]\n{err}")
                    continue
                    
            # Restart from scratch
            restarts_left -= 1
            attempt_num += 1
            if restarts_left < 0:
                vprint_section("COMPILATION FAILED - NO MORE RESTARTS")
                raise
            vprint(f"\n🔄 RESTART {max_restarts - restarts_left}/{max_restarts}")
            print("[restart] Starting a fresh attempt...")
            last_error_summary = ""


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("atomic_features", nargs='+', help="List of atomic feature filenames (e.g., grad_accum.py grad_norm_clip.py)")
    parser.add_argument("--model", type=str, default="anthropic/claude-3-5-sonnet-20240620", 
                        help="Provider/model string, e.g., 'openai/gpt-4o', 'anthropic/claude-3-5-sonnet-20240620'.")
    parser.add_argument("--api_key", type=str, default=None, help="Optional API key; otherwise use env vars.")
    parser.add_argument("--max_refine_attempts", type=int, default=3, help="Maximum number of refine attempts.")
    parser.add_argument("--max_restarts", type=int, default=3, help="Maximum number of restarts.")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output showing prompts, LLM responses, and detailed compilation progress.")
    args = parser.parse_args()

    # Set global verbose flag
    VERBOSE = args.verbose

    if VERBOSE:
        vprint_section("STARTING LLM TRAIN LOOP COMPILER")
        vprint(f"Model: {args.model}")
        vprint(f"API Key: {'***' if args.api_key else 'from environment'}")
        vprint(f"Max refine attempts: {args.max_refine_attempts}")
        vprint(f"Max restarts: {args.max_restarts}")
        vprint(f"Atomic features: {args.atomic_features}")

    llm = create_llm(args.model, api_key=args.api_key)
    compile_loop(
        args.atomic_features, 
        llm=llm, 
        max_refine_attempts=args.max_refine_attempts, 
        max_restarts=args.max_restarts
    )