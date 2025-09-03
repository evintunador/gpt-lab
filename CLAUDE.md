# GPT-Lab Repository

This is the root directory of a PyTorch research repository focused on helping ML researchers create modular, testable code and highly reproducible experiments.

## Directory Structure

- `train_loops/` - Catalog of training loops and tools relevant to training loops. Its primary notable feature consists of LLM-driven "compilation" by presenting different "atomic feature" training loop examples and having the LLM write a loop with all of the individual features combined. Researchers are also welcome to write their own loops manually. Planned features include 1) bulk testing of all training loops, 2) optional feature-specific tests and 3) an API that infers the user's intended training loop behavior and compiles it (or fetches from cache if already available)
- `modules/` - Catalog of `nn.Module`'s with bulk testing and bulk runtime/memory benchmarking frameworks. The structure is designed to allow for the integration of new `nn.Module`'s into the testing & benchmarking frameworks to be entirely optional
- `utils/` - Shared utilities across the project
    - `llm_code_compiler/` - LLM interface for code generation. Currently used to "compile" individual "atomic feature" training loops into multi-feature versions
    - `reproducibility/` - Tools for reproducible research (planned)
    - `distributed.py` - Tools to make other scripts agnostic single gpu vs torchrun vs slurm (planned)
- `data_sources/` - Data loading and preprocessing components
- `models/` - Catalog of models. Might in the future create tests specific to different types of models (llms, lvlms, autoencoders, etc)
- `optimizers/` - Catalog of optimizers. Includes a simple benchmarking tool.
- `benchmarks/` - Catalog of benchmarks. Defines different benchmark types which allows for easily swappable benchmark datasets and plugging in of models (planned)
- `experiments/` - Catalog of experiment workflows. Utilizes `utils/reproducibility/` to ensure reproducibility.

## Getting Started

1. Set up environment: `bash setup_env.sh`
2. Install dependencies: `pip install -r requirements.txt`
3. Run tests: `pytest`
4. Start adding to the various component catalogues.
5. Write and run your experiment over in `experiments/`

# Training Loop Testing Framework Implementation Instructions
## Project Overview
Build a comprehensive bulk testing framework for training loops in the train_loops/ directory. The system should automatically test all training loops (atomic features, LLM-compiled combinations, and custom loops) for correctness, and integrate with the existing LLM compiler.
## Directory Structure and Components
### Files to Create/Modify
- train_loops/bulk_tests.py - Main testing framework (NEW)
- utils/testing.py - Shared testing utilities (POPULATE EMPTY FILES)
- train_loops/llm_train_loop_compiler.py - Refactor to use shared testing (MODIFY)
### Files to Reference
- train_loops/catalog/atomic_features/base_loop.py - Baseline performance reference
- modules/bulk_test.py - Existing testing patterns to follow. These were built for nn.Module testing which is different in that every single nn.Module is different and therefore needs different tests, whereas training loops have one thing in common (that a model's loss should go down) and therefore can undergo one standardized test. However, they should also be able to go through individualized tests as well.
- modules/base_test_bench_utils.py - Utilities to potentially move/reuse
## Requirements
### Core Functionality
- Auto-Discovery: Automatically find all .py files in:
    - train_loops/catalog/atomic_features/
    - train_loops/catalog/llm_compiled/
    - train_loops/catalog/custom/ (this folder is where human experimenters can manually write custom training loops)
- Loop Validation: For each discovered file:
    1. Check if run_training() function exists (skip if not)
    2. Validate function signature matches: run_training(model, optimizer, loss_fn, train_loader, **kwargs) -> dict
    3. Test that function actually improves loss
    4. Learning Test: Use exact same test as current _universal_learning_test():
        - Synthetic binary classification: 2048x32 input → binary output
        - Simple MLP: Linear(32,64) → ReLU → Linear(64,2)
        - AdamW optimizer (lr=3e-3)
        - CrossEntropyLoss
        - Must achieve ≥10% loss reduction (post < pre * 0.9)
    5. Performance Baseline: Compare against base_loop.py:
        - Training time must be within 1 order of magnitude
        - GPU memory consumption must be within 1 order of magnitude (if cuda; mps doesn't provide an easy way to monitor memory)
- Device Support:
    - Use best_device from utils.py by default
    - Support pytest device override: pytest --device cpu
- Determinism: During testing only:
    - Set torch.manual_seed(0)
    - Enable CUDA deterministic if available: torch.backends.cudnn.deterministic = True
    - Do NOT force determinism in training loops themselves
### Integration Requirements
LLM Compiler Integration:
- Move _universal_learning_test() from llm_train_loop_compiler.py to test_utils/
- LLM compiler imports and calls the same test function
- Maintain existing caching and compilation behavior
Pytest Integration:
- Single test file that pytest discovers
- Use parameterized tests for different devices (if needed later)
- Clear pass/fail reporting with file paths
## Design Decisions
### Testing Strategy
- Black Box Testing: Only test inputs/outputs, no internal state inspection
- Synthetic Data Only: Use the existing simple classification task
- Existing Combinations Only: Only test LLM-compiled loops that already exist
- No Error Tolerances: Pass/fail based on loss reduction only
- Auto-Discovery: No manual test configuration required
### Error Handling
- Graceful Skipping: Skip files without run_training() function
- Clear Error Messages: Report file paths and specific failure types
- Import Failures: Handle and report module import errors
- Signature Validation: Check function signatures before testing
### Performance and Caching
- No Test Caching: Run all tests every time (they're fast)
- Baseline Comparison: Use base_loop.py as performance reference