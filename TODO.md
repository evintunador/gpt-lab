

# todo
## important / urgent

- [ ] do first experiment ([[DAGSeq2DAGSeq]])
    - [ ] dataset
        - [x] wiki dump parser
        - [x] graph structure constructor
        - [ ] torch dataset object
    - [ ] data loader
    - [ ] data sampler
        - [ ] factory
        - [ ] graph straversal strategy interface
        - [ ] different graph traversal algorithms
            - [ ] depth-first
            - [ ] breadth-first
            - [ ] random walk
            - [ ] composite graph traversal algorithm wrapper
    - [ ] model
        - [ ] base model (ripped from modded nanogpt or my own flavor)
        - [ ] flex attention mask
        - [ ] sequence-parallel???
    - [ ] reassess what we need after having actually used this system in DAGSeq2DAGSeq
- reproducibility manager:
    - [ ] turn `is_main_process` back into an argument
    - [ ] also check for git submodules & supermodules
    - [ ] make sure every one of the info giving functions is actually being called & saved into the folder. pretty sure rn some are not
- logger.py
    - [ ] draw a clearer conceptual distinction between logs (unstructured output) and metrics (structured output)
    - [ ] can i simplify all these overly complicated attempts somehow? maybe just forcibly intercept all print() statements so that dumb ML engineers end up always logging by accident? idk, this sdtout/err thing is killing me

### important / not-urgent

- [ ] design & build a mu-parametrization utility
- [ ] design and build a system for comparing performance between two experiments or/
and i guess different config settings within an experiment both directly and as a function of the performance per runtime/memory difference
    - [x] minimal proof of concept
    - [ ] time series
    - [ ] more adaptable to whatever's available in the experiments
- [ ] add slurm capabilities to DistributedManager
- [ ] build a tool to allow the experiment to dynamically increase or decrease the number of nodes it's taking up by periodically checking for outside requests. it'd have to effectively re-adjust gradient accumulation settings in order to make the experiment numerically equivalent to when it had more/fewer nodes. i guess it wouldn't have to be aware of VRAM utilization since we'd keep the micro batch size the same and only change number of nodes and number of gradient accumulation steps? i don't think this would have to be aware of the gradient accumulation atomic feature; you'd just need to tell it which argument is the right one. also this would have to overwrite whatever "waiting in line" system submitit has going on in order to restart a given experiment but smaller and let it skip forward in line. also ugly af thinking about resuming from the most recent checkpoint ew. not sure how feasible this is but i feel like it's necessary eventually. yeah it'd basically have to be a daemon that monitors for open nodes and alerts scripts that a node is available, which they check for every so often during training iterations, and if they get it then they get to shut themselves down and scale themsevles up. then when it comes to reducing jobs, i'd like a command like `sreduce` for people to manually use that writes to a file or something, and then that job checks said file for if a request to reduce itself is up. under this system it basically has to be the case that people play fair and only do it while there's nobody waiting in line, or only one person waiting in line and that job is going to take up the exact number of nodes that are being freed up. need some kind of guarantee to assure that the resume from checkpoint new job doesn't just get stuck at the back of the line
- [ ] implement more advanced parallel abilities for `src/gpt_lab/nn_modules/` testing and benchmarking and general utils to help with the various types of parallelization, maybe in DistributedManager? maybe in its own ParallelizationManager?
- [ ] make the tests inside an experiment that don't interact with gpt-lab not need to use `from experiments.<experiment_name>...`
- [ ] make the non-gpt_lab-related tests inside an experiment not get picked up as part of other experiments when running `CLIs/pytest_all_experiments.py`

### not important / urgent
- lmao this category basically doesn't exit

### not important / not urgent

- [ ] design & build hyperparameter search utility with an interface such that we can change out search algorithms later. will use the current multi-run utility
- [ ] setup a docker container to develop in to ensure consistent behavior across systems
- [ ] fact check various inaccuracies in the documentation
- [ ] figure out a way to combine hyperparameter search, mu-parameterization, and model size & gpu vram awareness to allow for an automated model scale up. this might be asking too much
- [ ] build a profiling system, likely for experiments themselves since what you care about at the end of the day is the full training loop's speed. hopefully i can use pytorch's built-in (it has one right?)
- [ ] find a cooler name for the repo
    - posture (bc it's helping you keep "good posture" when doing experiments)
    - {ml/dl/research/experiment/?}_harness
    - {ml/dl/?}-lab
    - just "lab"?
- [ ] design and build a wrapper around other general experiment utilities that need to be called at initialization to make the repo easier to use? do we even have enough stuff for that to be worth it? something like `ExperimentInitializer`
- [ ] in llm train loop compiler, setup a "this atomic feature is a superset of atomic feature x" system that saves some context length for the LLM which should hopefully help both performance and costs
- [ ] abstract out some of what's in `src/gpt_lab/train_loops/` into `src/gpt_lab/llm_code_compiler/` and find other use cases for our llm compiler system
- [ ] go around the repo looking for shared utilities across different catalog types that can be abstracted out
- [ ] revisit older project components that may have not been designed optimally (I'm particularly thinking of `src/gpt_lab/nn_modules/`)
- [ ] reduce duplicate dependencies (eg. we have both plotly & matplotlib)
- [ ] configuration.py 
    - [ ] make it not require any input argparser at all so users can rely entirely on config if they want
    - [ ] make an object that can be accessed equally as well like a dict or with methods or with string of methods. like `cfg.model.embed_dim` and `cfg['model']['embed_dim']` and `cfg['model.embed_dim']` all do the same thing. all the while still being serializable with a pretty print or a json. and even `cfg['embed_dim']` which will work if there is no other `'embed_dim'` key anywhere in the nested dict structure or throw an error otherwise. might even force non-duplicate keys throughout the nested dict structure. this one actually sounds super fun to make from a data structures perspective ngl
- [ ] make the example submodule experiment an actual submodule. i mean dagseq2dagseq already is so no point really
- [ ] edit GLU to use our own custom version of LigerKernel that supports more activation functions
- [ ] get rid of the nn_module bulk test's backup test discovery system
- [ ] improve attribute names and standardize all test & bench configs (hint: rename 'output_validator' to 'output_validator_fn' in nn.Modules bulk testing)
- [ ] write equivalents of `to_device` and `to_dtype` but for `.clone()`, `.detach()`, `.contiguous()`, etc. maybe even abstract it out for fun. i guess it's just a recursive type-aware apply huh. sounds dumb now that i say it that way
- [ ] build a schema tracking system for updating a model's checkpoints as they change over time. like hash the combination keys, shapes, and dtypes and keep track of existing conversion functions that we have in a registry of some sort
- [x] move reproducibility's storage backup backend ABC & daemon ABC over to the catalog system
- multi-run
    - [ ] add non-grid indivdiual-command ability
    - [ ] figure out how to make it aware of when a run ends. have it be a process acting as a daemon monitoring some file or something? have it monitor what files those processes are _writing_ to and set a timeout to trigger of said processes stop writing to their folders? monitor said files for a file they should be printing upon __exit__ in order for them to be compatible? infer said folder the process is working in as the lowest folder that that process has written (not read) to?
- [ ] develop pre-tokenizing/caching as its own catalog system
    - [x] separate the IO ops from the llm squential tokens logic
    - [ ] geenralize the IO class to different dtypes
    - [ ] do something more intelligent than the 11041999 doc prefix signature. maybe each doc prefix is a hash of its own code or something? or just test requirement that each implementation in the system write a different prefix?
- [x] move a couple reproducibility-related method from logger.py to reproducibility.py