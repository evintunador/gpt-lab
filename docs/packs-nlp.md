# NLP Pack

The NLP pack (`catalogs/packs/nlp/gpt_lab/`) provides components specifically for natural language processing experiments.

## Overview

The NLP pack includes:
- **benchmarks**: Evaluation benchmarks for language models
- **data_sources**: Datasets for pretraining and evaluation
- **models**: Model components like tokenizers

## Activation

To use the NLP pack, activate it in your experiment:

```yaml
# experiments/my_nlp_exp/gpt_lab.yaml
include_experiments: []
include_packs: ['nlp']
```

Or via environment variable:

```bash
export GPT_LAB_ACTIVE_PACKS=nlp
```

## Testing

Test NLP pack components:

```bash
# Activate NLP pack
export GPT_LAB_ACTIVE_PACKS=nlp

# Run tests
pytest src/gpt_lab/tests/ -v
```

## Benchmarking

Benchmark NLP models:

```bash
# Activate pack
export GPT_LAB_ACTIVE_PACKS=nlp

# Run benchmarks
python -m gpt_lab.benchmarks.runner \
    --model my_model.pt \
    --benchmarks hellaswag,wiki_qa,asdiv \
    --output results.json
```

## File Structure

```
catalogs/packs/nlp/gpt_lab/
├── benchmarks/
│   ├── fill_in_the_blank.py
│   └── multiple_choice.py
│   └── ...
├── data_sources/
│   ├── pretraining/
│   │   └── fineweb.py
│   │   └── ...
│   └── benchmarks/
│       ├── fill_in_the_blank/
│       │   └── asdiv.py
│       │   └── ...
│       └── multiple_choice/
│           ├── hellaswag.py
│           └── wiki_qa.py
│           └── ...
├── models/
│   └── tokenizers/
│       └── tiktoken.py
│       └── ...
└── artifacts/
```
