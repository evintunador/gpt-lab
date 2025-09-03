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