# GPT-Lab Repository

This is the root directory of a PyTorch research repository focused on helping ML researchers create modular, testable code and highly reproducible experiments. 
It's purpose is to be a holding place of catalogs of different core components of an experiment (eg. training loops, `nn.Module`'s, etc.) in use by a given research group. 
It was designed with the ability to have some participants completely ignore the many features of the harness if they want, and compliance to generally be easy to add on later. 
Each catalog has some combination of unit tests and/or performance benchmarks that vary in their capabilities depending on the particulars of what that directory is cataloging. 

## Directory Structure

- `docs/` - A full mirror of the `src/` repo with documentation about the user-facing API of each corresponding file (planned)
- `experiments/` - Catalog of experiment workflows. Utilizes `utils/reproducibility/` to ensure reproducibility. (planned)
- `src/`

    - `train_loops/` - catalog of training loops. its primary notable feature consists of LLM-driven "compilation" by presenting different "atomic feature" training loop examples and having the LLM write a loop with all of the individual features combined. Researchers are also welcome to write their own loops manually. Also includes 1) bulk testing of all training loops, 2) optional feature-specific tests and 3) an API that infers the user's intended training loop behavior and compiles it (or fetches from cache if already available)
    - `modules/` - catalog of `nn.Module`'s with bulk testing and runtime/memory benchmarking frameworks.
    - `utils/` - shared utilities across the project

        - `llm_code_compiler/` - llm interface for code generation. currently used to "compile" individual "atomic feature" training loops into multi-feature versions, but hoping more parts of the repo will find it useful later
        - `reproducibility/` - tools for reproducible research (planned)
        - `distributed.py` - tool to make scripts agnostic to whether they're being run with single gpu vs torchrun vs slurm. recommended pattern for use is `with DistributedManager() as manager: ...`
        - `checkpointing.py` - simple tool for saving & loading metadata & objects with state dicts. should be general enough to work with any weird training setup and allow for resuming training right where it left off.
        - `device.py`
        - `benchmarking.py`
        - `testing.py`

    - `data_sources/` - data loading and preprocessing components
    - `models/` - catalog of models
    - `optimizers/` - catalog of optimizers. includes simple testing and benchmarking tools.
    - `benchmarks/` - catalog of benchmarks. defines different benchmark types which allows for easily swappable benchmark datasets and plugging in of models

## Getting Started

1. Set up environment: `bash setup_env.sh`
2. Install dependencies: `pip install -r requirements.txt`
3. Run tests: `pytest`
4. Start adding to the various component catalogues.
5. Write and run your experiment over in `experiments/`

## todo
### important / urgent

- [ ] do first DAGSeq2DAGSeq experiment

    - [ ] reassess what we need after having actually used this system in DAGSeq2DAGSeq

- [x] bulid tools for model checkpointing saving & loading
- [x] design and build a logging system
- [ ] design and build a system for comparing performance between two experiments or/and i guess different config settings within an experiment (presumably benchmark outputs need to be saved in experiment directory) both directly and as a function of the performance per runtime/memory difference
- [ ] design and build reproducibility tools that force a git commit maybe at an independent branch or something for every experiment. independent branch might get messy idk; i guess it might make more sense to propose a more linear history. i'll figure all that out when i get there.

### important / not-urgent
- [ ] design & build hyperparameter search utility with an interface such that we can change out search algorithms later. this will be used to inform experiments. design & build a mu-parametrization utility to be used in experimentation. design & build a system that does the former and then utilizes its results to inform the latter when running experiments. maybe like does hyperparameter search at small scale, uses those results to rank choices for hyperparameters of big scale model, and then from those choices goes down the list of priority until one fits into gpu memory, and then runs that for real. obvi needs to incorporate mu parametrization
- [ ] add FSDP as an ability somewhere in here, not sure where
- [ ] find other desirable features to help with our experiments
- [ ] implement more advanced parallel abilities for [`modules/`](src/modules/) testing and benchmarking and general utils to help with the various types of parallelization

### not important / urgent


### not important / not urgent

- [ ] add more atomic feature training loops
- [ ] find other use cases for our llm compiler system & abstract out some of what's in the [`train_loops/`](src/train_loops/catalog_llm_compiler.py) one into shared utilities.
- [ ] add more benchmark datasets
- [ ] go around the repo looking for shared utilities across different catalog types that can be abstracted out and put into `utils/`
- [ ] tool for forking repo with specific experiment as the only one to carry over into fork--or i guess a tool to run after you've forked? not sure how the system will work. maybe just a simple tool that, after a fork, you give it the directories inside `experiments/` that you actually care about, and it deletes all catalog items that are not used by those experiments? or, optionally, also deletes all harness component files that weren't utilized. or, even more optionally, also deletes any functions and classes within the remaining files that weren't used? not sure exactly how i'd properly parse that dependency graph but i assume it's doable.
- [ ] revisit older project components that may have not been designed optimally (I'm particularly thinking of `mdoules/`)
