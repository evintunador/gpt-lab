# GPT-Lab repository

This is a repository focused on helping ML researchers (both individuals and organizations) create modular, testable code and highly reproducible experiments while facilitating rapid iteration. 
It was designed to impose minimal structure, meaning researchers may completely ignore many of features of the harness if they want, and compliance is generally be easy to add on post-hoc. 
Each catalog has some combination of unit tests and/or performance benchmarks that vary in their capabilities depending on the particulars of what that directory is cataloging, and some have extra interesting features. 

## directory structure

### top-level directories
- `src/gpt_lab/` - main package source code (see below for details)
- `experiments/` - catalog of experiment workflows. there are a bunch of common and useful examples already in there; go build your own
- `docs/` - documentation about the user-facing API of each corresponding file (planned)
- `bench_results/` - benchmark results and performance data
- `tools/` - utility scripts for viewing benchmark results and other tools

### main package (`src/gpt_lab/`)
- `benchmarks/` - catalog of benchmarks. defines different benchmark types which allows for easily swappable benchmark datasets and plugging in of models
- `catalog_utils.py` - shared utilities for catalog management
- `checkpointer.py` - tool for saving & loading metadata & objects with state dicts. should be general enough to work with any weird training setup and allow for resuming training right where it left off.
- `configuration.py` - tool for combining `.yaml` configs with terminal arguments
- `data_sources/` - data loading and preprocessing components
- `device.py` - simple tools for device management
- `distributed.py` - tool to make scripts agnostic to whether they're being run with single gpu vs torchrun vs slurm. recommended pattern for use is `with DistributedManager() as manager: ...`
- `llm_code_compiler/` - llm interface for automated and testable code generation. currently used to "compile" individual "atomic feature" training loops into multi-feature versions, but hoping more parts of the repo will find it useful later. if not, then it'll be folded into `train_loops/`
- `logger.py` - tool for structured logging
- `models/` - catalog of models. the distinction between a model and a module lies in the latter being interfaceable with `.forward()` and the former being a wrapper with a variety of capabilities (eg. `.inference()`, `.batched_inference()`, `.benchmark()`, `.get_attention_logits()`, etc.)
- `nn_modules/` - catalog of `nn.Module`'s with bulk testing and runtime/memory benchmarking frameworks.
- `optimizers/` - catalog of optimizers. includes testing and benchmarking tools.
- `reproducibility.py` - tools for reproducible research
- `train_loops/` - catalog of training loops. its primary notable feature consists of LLM-driven "compilation" by presenting different "atomic feature" training loop examples and having the LLM write a loop with all of the individual features combined. Researchers are also welcome to write their own loops manually. Also includes 1) bulk testing of all training loops, 2) optional feature-specific tests and 3) an API that infers the user's intended training loop behavior and compiles it (or fetches from cache if already available)


## getting started

1. Install the project in editable mode. For development, it's recommended to install with all catalog and testing dependencies:
   ```bash
   pip install -e '.[dev]'
   ```
   If you only need core dependencies, you can run `pip install -e .`. To install dependencies for a specific catalog, use `pip install -e '.[<catalog_name>]'` (e.g., `pip install -e '.[optimizers]'`). To install all catalog dependencies at once, use `pip install -e '.[all_catalogs]'`.
   
   **Note:** The quotes around `.[...]` are required for zsh users (default shell on macOS) to prevent glob expansion.

2. run tests to confirm it all works: `pytest`
3. add to the various component catalogues.
4. write and run your experiment over in `experiments/`

## todo
### important / urgent

- [ ] implement nanogpt & moddednanogpt as experiment for demonstration purposes. i'm sure eventually i'll fill in other common models into the catalog as well
    - [x] first-draft implemented but nothing has been run/tested yet since i'm working off a mac and flex attention requires cuda
- [ ] abstract out evaluation utilities. rn we've got `src/benchmarks/` which seems able to run benchmark datasets but i'd also like general evaluation metrics like perplexity to get recorded.
- [ ] design and build a system for comparing performance between two experiments or/and i guess different config settings within an experiment both directly and as a function of the performance per runtime/memory difference
- [ ] build a profiling system, likely for experiments themselves since what you care about at the end of the day is the full training loop's speed
- [ ] do first DAGSeq2DAGSeq experiment

    - [ ] reassess what we need after having actually used this system in DAGSeq2DAGSeq

### important / not-urgent

- [x] Refactor the project structure to use a `src`-layout. This will simplify package discovery, eliminate the need to manually update `pyproject.toml` when new top-level modules/packages are added, and align with modern Python packaging standards.
- [ ] design & build hyperparameter search utility with an interface such that we can change out search algorithms later. this will be used to inform experiments. design & build a mu-parametrization utility to be used in experimentation. design & build a system that does the former and then utilizes its results to inform the latter when running experiments. maybe like does hyperparameter search at small scale, uses those results to rank choices for hyperparameters of big scale model, and then from those choices goes down the list of priority until one fits into gpu memory, and then runs that for real. obvi needs to incorporate mu parametrization
- [ ] add FSDP as an ability somewhere in here, not sure where. i guess a feature of DistributedManager?
- [ ] add slurm capabilities to DistributedManager
- [ ] implement more advanced parallel abilities for `src/gpt_lab/nn_modules/` testing and benchmarking and general utils to help with the various types of parallelization, maybe in DistributedManager?

### not important / urgent


### not important / not urgent

- [ ] find a cooler name for the repo
- [ ] design and build a wrapper around other general experiment utilities to make the repo easier to use? do we even have enough stuff for that to be worth it? it's really just the two with statements rn
- [ ] add more atomic feature training loops
- [ ] setup a "this atomic feature is a superset of atomic feature x" system that saves some context length for the LLM which should hopefully help both performance and costs
- [ ] abstract out some of what's in `src/gpt_lab/train_loops/` into `src/gpt_lab/llm_code_compiler/` and find other use cases for our llm compiler system
- [ ] add more benchmark datasets
- [ ] go around the repo looking for shared utilities across different catalog types that can be abstracted out
- [ ] tool for forking repo with specific experiment as the only one to carry over into fork--or i guess a tool to run after you've forked? not sure how the system will work. maybe just a simple tool that, after a fork, you give it the directories inside `experiments/` that you actually care about, and it deletes all catalog items that are not used by those experiments? or, optionally, also deletes all harness component files that weren't utilized. or, even more optionally, also deletes any functions and classes within the remaining files that weren't used? not sure exactly how i'd properly parse that dependency graph but i assume it's doable.
- [ ] revisit older project components that may have not been designed optimally (I'm particularly thinking of `src/gpt_lab/nn_modules/`)
- [ ] really a loss function as input to a training loop is an assumption rather than always being the case since someone could fold the loss function into the `nn.Module` itself. edit all the atomic features to remove this as an assumption and create a `loss_function.py` atomic feature
- [ ] reduce duplicate dependencies (eg. we have both plotly & matplotlib)
- [x] switch tests from current next-to-file setup to one where each catalog directory gets its own test directory (eg. `src/gpt_lab/nn_modules/tests/` and `src/gpt_lab/train_loops/tests/`). will require editing pyproject.toml to tell pytest that there are multiple tests/ folders. better than one single tests/ folder since it's more in line with our modular catalog design
