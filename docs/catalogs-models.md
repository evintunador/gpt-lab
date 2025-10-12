# Models Catalog

High-level model wrappers and utilities that extend beyond `nn.Module`.

## Overview

The models catalog contains classes that wrap or extend `nn.Module` functionality for specific use cases like inference, generation, or evaluation.

**Namespace**: `gpt_lab.models`

## Purpose

Models in this catalog:
- Are NOT simple `nn.Module` subclasses (those go in `nn_modules`)
- Wrap one or more `nn.Module` instances
- Provide task-specific interfaces (inference, generation, evaluation)
- Integrate with benchmarks via decorators

## Example: Tokenizers

The NLP pack provides tokenizers in `gpt_lab.models.tokenizers`:

```python
from gpt_lab.models.tokenizers import TiktokenTokenizer

tokenizer = TiktokenTokenizer(encoding_name="gpt2")

# Encode text
tokens = tokenizer.encode("Hello, world!")

# Decode tokens
text = tokenizer.decode(tokens)

# Get vocabulary size
vocab_size = tokenizer.vocab_size
```

## Example Structure

A typical model wrapper:

```python
# experiments/my_exp/gpt_lab/models/my_llm.py

import torch.nn as nn

class MyLLM:
    """LLM wrapper with inference and generation methods."""
    
    def __init__(self, backbone: nn.Module, tokenizer):
        self.backbone = backbone
        self.tokenizer = tokenizer
    
    def inference(self, text: str) -> dict:
        """Single text inference."""
        tokens = self.tokenizer.encode(text)
        outputs = self.backbone(tokens)
        return {"logits": outputs}
    
    def batched_inference(self, texts: list[str]) -> dict:
        """Batched inference."""
        # Implementation
        pass
    
    def generate(self, prompt: str, max_length: int = 100) -> str:
        """Generate text from prompt."""
        # Implementation
        pass
```

## Integration with Benchmarks

Models can define benchmark-compatible methods:

```python
from gpt_lab.benchmarks import benchmark_method

class MyLLM:
    @benchmark_method("multiple_choice")
    def answer_multiple_choice(self, question, choices):
        """Answer multiple choice question."""
        # Implementation
        return predicted_choice
    
    @benchmark_method("fill_in_blank")
    def predict_masked(self, text_with_mask):
        """Predict masked token."""
        # Implementation
        return prediction
```

## Contributing

Add models to:
- Pack: `catalogs/packs/<pack>/gpt_lab/models/`
- Experiment: `experiments/<exp>/gpt_lab/models/`

Models should provide clear interfaces for their intended use cases.
