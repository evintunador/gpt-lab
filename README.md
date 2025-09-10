# GPT-Lab Repository

This is the root directory of a PyTorch research repository focused on helping ML researchers create modular, testable code and highly reproducible experiments. 
It's purpose is to be a holding place of catalogs of different core components of an experiment (eg. training loops, `nn.Module`'s, etc.) in use by a given research group. 
It was designed with the ability to have some participants completely ignore the many features of the harness if they want, and compliance to generally be easy to add on later. 
Each catalog has some combination of unit tests and/or performance benchmarks that vary in their capabilities depending on the particulars of what that directory is cataloging. 

## Directory Structure

- `docs/` - A full mirror of the `src/` repo with documentation about the user-facing API of each corresponding file (planned)
- `src/`
    - `train_loops/` - Catalog of training loops. Its primary notable feature consists of LLM-driven "compilation" by presenting different "atomic feature" training loop examples and having the LLM write a loop with all of the individual features combined. Researchers are also welcome to write their own loops manually. Also includes 1) bulk testing of all training loops, 2) optional feature-specific tests and 3) an API that infers the user's intended training loop behavior and compiles it (or fetches from cache if already available)
    - `modules/` - Catalog of `nn.Module`'s with bulk testing and runtime/memory benchmarking frameworks.
    - `utils/` - Shared utilities across the project
        - `llm_code_compiler/` - LLM interface for code generation. Currently used to "compile" individual "atomic feature" training loops into multi-feature versions, but hoping more parts of the repo will find it useful later
        - `reproducibility/` - Tools for reproducible research (planned)
        - `distributed.py` - Tools to make other scripts agnostic to single gpu vs torchrun vs slurm (planned)
        - `device.py`
        - `benchmarking.py`
        - `testing.py`
    - `data_sources/` - Data loading and preprocessing components
    - `models/` - Catalog of models. 
    - `optimizers/` - Catalog of optimizers. Includes simple testing and benchmarking tools.
    - `benchmarks/` - Catalog of benchmarks. Defines different benchmark types which allows for easily swappable benchmark datasets and plugging in of models (planned)
    - `experiments/` - Catalog of experiment workflows. Utilizes `utils/reproducibility/` to ensure reproducibility. (planned)

## Getting Started

1. Set up environment: `bash setup_env.sh`
2. Install dependencies: `pip install -r requirements.txt`
3. Run tests: `pytest`
4. Start adding to the various component catalogues.
5. Write and run your experiment over in `experiments/`

## TODO

- [x] designa and build a user-facing API for the [`train_loops/](src/train_loops/) folder that intelligently picks which training loop features to use based on input arguments
    - [ ] ~~remove the current behavior of allowing None to be passed in and not trigger that feature~~
- [x] add more atomic feature training loops
- [ ] build tools to allow experiment makers to be blissfully unaware of the difference between slurm vs torchrun vs single gpu, or at least get as close to that as possible
- [ ] bulid tools for model checkpointing
- [ ] make muon faster
- [ ] find other use cases for our llm compiler system & abstract out some of what's in the [`train_loops/`](src/train_loops/catalog_llm_compiler.py) one into shared utilities.
- [ ] design and build model accuracy benchmarking system where a benchmark type can be defined and models & datasets swapped out
- [ ] design and build reproducibility tools that force a git commit maybe at an independent branch or something for every experiment. independent branch might get messy idk; i guess it might make more sense to propose a more linear histry. i'll figure all that out when i get there.
- [ ] design & build hyperparameter search utility with an interface such that we can change out search algorithms later. this will be used to inform experiments. design & build a mu-parametrization utility to be used in experimentation. design & build a system that does the former and then utilizes its results to inform the latter when running experiments. maybe like does hyperparameter search at small scale, uses those results to rank choices for hyperparameters of big scale model, and then from those choices goes down the list of priority until one fits into gpu memory, and then runs that for real.
- [ ] add FSDP as an ability somewhere in here, not sure where
- [ ] find other features to help with our experiments
- [ ] go around the repo looking for shared utilities across different catalog types that can be abstracted out and put into `utils/`
- [ ] do first DAGSeq2DAGSeq experiment
- [ ] reassess what we need after having actually used this system in DAGSeq2DAGSeq
- [ ] implement tensorparallel abilities for [`modules/`](src/modules/) testing and benchmarking (low priority)
- [ ] auto preprint latex generator (low priority, if ever)