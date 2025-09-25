# Inspired by discussions similar to
# https://medium.com/redsquirrel-tech/llm-as-compiler-2a2f79d30f0b

import sys
import traceback
import inspect
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable

from gpt_lab.train_loops.tests.test_train_loops_catalog import (
    universal_learning_test, discover_specific_tests, base_loop_compliance_test, dataset_type_compatibility_test
)
from gpt_lab.llm_code_compiler import LLMClient, create_llm
from gpt_lab.device import get_default_device
from gpt_lab.catalog_utils import import_module_from_path

# Get a logger for this module
logger = logging.getLogger(__name__)


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
- Ensure compatibility with bost map-style and iterable-style datasets. Never call `len()` on a dataloader.

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
    base_loop_path = Path(__file__).parent / "catalog" / "atomic_features" / "base_loop.py"
    
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


def run_specific_tests_for_compilation(run_training_fn: Callable, atomic_features: List[str], device: str):
    """Run all applicable specific tests during compilation - validation only."""
    specific_tests = discover_specific_tests()
    
    for feature in atomic_features:
        clean_feature = feature.replace('.py', '')  # Handle both formats
        if clean_feature in specific_tests:
            for test_func in specific_tests[clean_feature]:
                # Just run the test - if it fails, compilation fails
                test_func(run_training_fn, device)


def run_base_loop_compliance_test_for_compilation(run_training_fn: Callable, atomic_features: List[str], device: str):
    """Run base_loop compliance test during compilation."""
    
    # For compiled loops, we use a representative name from the atomic features
    feature_name = f"compiled_loop_{'-'.join(sorted([f.replace('.py', '') for f in atomic_features]))}"
    base_loop_compliance_test(run_training_fn, feature_name, device)


def compile_loop(
    atomic_features: List[str],
    llm: Optional[LLMClient] = None,
    max_refine_attempts: int = 3, 
    max_restarts: int = 3,
) -> Dict[str, Any]:
    """
    Main entry: ask LLM for a bespoke training loop combining atomic features, test it, cache it.
    
    Note: This function is designed for combining multiple atomic features. For single features,
    use the atomic feature directly for better performance and reliability.
    """
    # Validate input
    if not atomic_features:
        raise ValueError("No atomic features provided for compilation")
    
    if len(atomic_features) == 1:
        atomic_path = Path(__file__).parent / "catalog" / "atomic_features" / f"{atomic_features[0]}.py"
        raise ValueError(
            f"Compilation not recommended for single atomic feature '{atomic_features[0]}'. "
            f"Use the atomic feature directly instead for better performance. "
            f"File: {atomic_path}.py"
        )
    
    llm = llm or LLMClient()
    name = _make_descriptive_name(atomic_features)
    code_path = Path(__file__).parent / "catalog" / "llm_compiled" / f"{name}.py"
    device = get_default_device()

    logger.debug("=" * 60)
    logger.debug("LLM TRAINING LOOP COMPILATION")
    logger.debug("=" * 60)
    logger.debug(f"Atomic features: {atomic_features}")
    logger.debug(f"Generated name: {name}")
    logger.debug(f"Output path: {code_path}")

    # Cached success path
    if code_path.exists():
        logger.debug("-" * 40)
        logger.debug("CACHE CHECK")
        logger.debug("-" * 40)
        logger.debug(f"Found existing file at {code_path}")
        try:
            module = import_module_from_path(f"cached_loop", code_path)
            logger.debug("✓ Successfully loaded cached loop")
            logger.info(f"Using cached compiled loop at {code_path}")
            return {
                "name": name, 
                "code_path": str(code_path), 
                "atomic_features": atomic_features,
            }
        except Exception as e:
            logger.debug(f"✗ Failed to load cached loop: {e}")
            logger.warning(f"Failed to load cached loop {code_path}: {e}")
            logger.info("Will regenerate...")

    # Build prompts
    logger.debug("-" * 40)
    logger.debug("PROMPT CONSTRUCTION")
    logger.debug("-" * 40)
    system_prompt = _build_system_prompt_with_base_loop()
    user_prompt = _build_user_prompt(atomic_features)
    
    logger.debug(f"System prompt:\n```\n{system_prompt}\n```")
    logger.debug(f"\nUser prompt:\n```\n{user_prompt}\n```")

    # Generate + refine loop
    restarts_left = max_restarts
    last_error_summary = ""
    code = ""
    attempt_num = 1
    
    logger.debug("-" * 40)
    logger.debug("CODE GENERATION AND VALIDATION")
    logger.debug("-" * 40)
    
    while restarts_left >= 0:
        try:
            if not last_error_summary:
                logger.debug(f"\n🚀 ATTEMPT {attempt_num}: Initial generation")
                logger.debug("Calling LLM.generate()...")
                code = llm.generate(system_prompt, user_prompt)
            else:
                logger.debug(f"\n🔄 ATTEMPT {attempt_num}: Refinement")
                logger.debug("Calling LLM.refine()...")
                logger.debug(f"Error to fix: {last_error_summary}")
                code = llm.refine(system_prompt, user_prompt, prior_code=code, error_summary=last_error_summary)
            
            logger.debug("✓ LLM response received")
            logger.debug(f"Generated code length: {len(code)} characters")
            logger.debug(f"Generated code preview:\n```python\n{code[:5000]}{'...' if len(code) > 5000 else ''}\n```")

            # Add metadata and write
            logger.debug("\n📝 Adding metadata and writing file...")
            code_with_metadata = _add_metadata_to_code(code, atomic_features, str(device))
            _write_file(code_path, code_with_metadata)
            logger.debug(f"✓ File written to {code_path}")
            
            logger.debug("\n🔍 Testing generated code...")
            
            # Import test
            logger.debug("Testing import...")
            try:
                module = import_module_from_path(f"compiled_loop", code_path)
                logger.debug("✓ Import successful")
            except Exception:
                err = _summarize_exception_filtered([str(code_path)], phase="[import]")
                logger.debug(f"✗ Import failed: {err}")
                raise RuntimeError(err)

            # Function signature test
            logger.debug("Checking for run_training function...")
            if not hasattr(module, "run_training"):
                logger.debug("✗ Missing run_training function")
                raise AssertionError("Generated file must define function 'run_training' with the required signature.")
            run_training_fn = getattr(module, "run_training")
            logger.debug("✓ run_training function found")

            # Universal test
            logger.debug("Running universal learning test...")
            try:
                universal_learning_test(run_training_fn, device=str(device))
                logger.debug("✓ Universal test passed")
            except Exception:
                err = _summarize_exception_filtered([str(code_path)], phase="[universal_test]")
                logger.debug(f"✗ Universal test failed: {err}")
                raise RuntimeError(err)

            # Dataset compatibility test
            logger.debug("Running dataset type compatibility test...")
            try:
                dataset_type_compatibility_test(run_training_fn, device=str(device))
                logger.debug("✓ Dataset type compatibility test passed")
            except Exception:
                err = _summarize_exception_filtered([str(code_path)], phase="[dataset_type_compatibility_test]")
                logger.debug(f"✗ Dataset type compatibility test failed: {err}")
                raise RuntimeError(err)

            # Base loop compliance test
            logger.debug("Running base_loop compliance test...")
            try:
                run_base_loop_compliance_test_for_compilation(run_training_fn, atomic_features, device=str(device))
                logger.debug("✓ Base loop compliance test passed")
            except Exception:
                err = _summarize_exception_filtered([str(code_path)], phase="[base_loop_compliance]")
                logger.debug(f"✗ Base loop compliance test failed: {err}")
                raise RuntimeError(err)

            # Specific tests
            logger.debug("Running specific feature tests...")
            try:
                run_specific_tests_for_compilation(run_training_fn, atomic_features, device=str(device))
                logger.debug("✓ All specific tests passed")
            except Exception:
                err = _summarize_exception_filtered([str(code_path)], phase="[specific_tests]")
                logger.debug(f"✗ Specific tests failed: {err}")
                raise RuntimeError(err)

            logger.info("=" * 60)
            logger.info("COMPILATION SUCCESSFUL")
            logger.info("=" * 60)
            logger.info(f"Compiled and validated. Cached at {code_path}")
            return {
                "name": name, 
                "code_path": str(code_path), 
                "atomic_features": atomic_features,
            }

        except Exception as e:
            # Pass only focused, phase-tagged errors into refine loop
            err = str(e)
            logger.debug(f"\n❌ ATTEMPT {attempt_num} FAILED: {err}")
            logger.error(f"Compilation/testing failed: {err}")
            
            # Try refine attempts first
            logger.debug(f"\n🔧 Starting {max_refine_attempts} refinement attempts...")
            for refine_attempt in range(max_refine_attempts):
                try:
                    logger.debug(f"\n🔧 REFINEMENT {refine_attempt + 1}/{max_refine_attempts}")
                    last_error_summary = err
                    logger.debug("Calling LLM.refine()...")
                    code = llm.refine(system_prompt, user_prompt, prior_code=code, error_summary=err)
                    logger.debug("✓ Refinement response received")
                    
                    code_with_metadata = _add_metadata_to_code(code, atomic_features, str(device))
                    _write_file(code_path, code_with_metadata)
                    logger.debug(f"✓ Refined code written to {code_path}")
                    
                    # Test refined code
                    logger.debug("Testing refined code...")
                    try:
                        module = import_module_from_path(f"compiled_loop", code_path)
                        logger.debug("✓ Import successful")
                    except Exception:
                        err = _summarize_exception_filtered([str(code_path)], phase="[import]")
                        logger.debug(f"✗ Import failed: {err}")
                        raise RuntimeError(err)
                        
                    if not hasattr(module, "run_training"):
                        logger.debug("✗ Missing run_training function")
                        raise AssertionError("Generated file must define function 'run_training'.")
                    run_training_fn = getattr(module, "run_training")
                    logger.debug("✓ run_training function found")
                    
                    try:
                        universal_learning_test(run_training_fn, device=str(device))
                        logger.debug("✓ Universal test passed")
                    except Exception:
                        err = _summarize_exception_filtered([str(code_path)], phase="[universal_test]")
                        logger.debug(f"✗ Universal test failed: {err}")
                        raise RuntimeError(err)
                    
                    try:
                        dataset_type_compatibility_test(run_training_fn, device=str(device))
                        logger.debug("✓ Dataset type compatibility test passed")
                    except Exception:
                        err = _summarize_exception_filtered([str(code_path)], phase="[dataset_type_compatibility_test]")
                        logger.debug(f"✗ Dataset type compatibility test failed: {err}")
                        raise RuntimeError(err)
                    
                    try:
                        run_base_loop_compliance_test_for_compilation(run_training_fn, atomic_features, device=str(device))
                        logger.debug("✓ Base loop compliance test passed")
                    except Exception:
                        err = _summarize_exception_filtered([str(code_path)], phase="[base_loop_compliance]")
                        logger.debug(f"✗ Base loop compliance test failed: {err}")
                        raise RuntimeError(err)
                        
                    try:
                        run_specific_tests_for_compilation(run_training_fn, atomic_features, device=str(device))
                        logger.debug("✓ All specific tests passed")
                    except Exception:
                        err = _summarize_exception_filtered([str(code_path)], phase="[specific_tests]")
                        logger.debug(f"✗ Specific tests failed: {err}")
                        raise RuntimeError(err)
                        
                    logger.info("=" * 60)
                    logger.info("REFINEMENT SUCCESSFUL")
                    logger.info("=" * 60)
                    logger.info(f"Successfully refined and cached at {code_path}")
                    return {
                        "name": name, 
                        "code_path": str(code_path), 
                        "atomic_features": atomic_features,
                    }
                except Exception as e2:
                    err = str(e2)
                    logger.debug(f"✗ REFINEMENT {refine_attempt + 1} FAILED: {err}")
                    logger.error(f"Refinement attempt failed: {err}")
                    continue
                    
            # Restart from scratch
            restarts_left -= 1
            attempt_num += 1
            if restarts_left < 0:
                logger.critical("=" * 60)
                logger.critical("COMPILATION FAILED - NO MORE RESTARTS")
                logger.critical("=" * 60)
                raise
            logger.warning(f"\n🔄 RESTART {max_restarts - restarts_left}/{max_restarts}. Starting a fresh attempt...")
            last_error_summary = ""


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("atomic_features", nargs='+', help="List of atomic feature filenames (e.g., grad_accum.py grad_norm_clip.py)")
    parser.add_argument("--model", type=str, default="anthropic/claude-sonnet-4-20250514", 
                        help="Provider/model string, e.g., 'openai/gpt-4o', 'anthropic/claude-sonnet-4-20250514'.")
    parser.add_argument("--api_key", type=str, default=None, help="Optional API key; otherwise use env vars.")
    parser.add_argument("--max_refine_attempts", type=int, default=3, help="Maximum number of refine attempts.")
    parser.add_argument("--max_restarts", type=int, default=3, help="Maximum number of restarts.")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output showing prompts, LLM responses, and detailed compilation progress.")
    args = parser.parse_args()

    # Configure logging for standalone script execution
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,  # Override any existing third-party logger configs
    )

    logger.info("=" * 60)
    logger.info("STARTING LLM TRAIN LOOP COMPILER (STANDALONE)")
    logger.info("=" * 60)
    logger.info(f"Model: {args.model}")
    logger.info(f"API Key: {'***' if args.api_key else 'from environment'}")
    logger.info(f"Max refine attempts: {args.max_refine_attempts}")
    logger.info(f"Max restarts: {args.max_restarts}")
    logger.info(f"Atomic features: {args.atomic_features}")

    llm = create_llm(args.model, api_key=args.api_key)
    compile_loop(
        args.atomic_features, 
        llm=llm, 
        max_refine_attempts=args.max_refine_attempts, 
        max_restarts=args.max_restarts
    )