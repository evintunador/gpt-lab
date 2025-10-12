# Benchmarks Catalog

Evaluation frameworks for assessing model performance.

## Overview

The benchmarks catalog provides standardized evaluation protocols for different tasks. Benchmarks are discovered across active roots and can be run on compatible models.

**Namespace**: `gpt_lab.benchmarks`

## Available Benchmarks

### NLP Pack Benchmarks

**Fill in the Blank**:
- Tests token prediction accuracy
- Example dataset: ASDiv (arithmetic reasoning)

**Multiple Choice**:
- Tests question answering ability
- Example datasets: HellaSwag, WikiQA

See `packs-nlp.md` for details.

## Using Benchmarks

```python
from gpt_lab.benchmarks import MultipleChoiceBenchmark

# Create benchmark
benchmark = MultipleChoiceBenchmark(
    dataset_name="hellaswag",
    num_samples=1000
)

# Run evaluation
results = benchmark.evaluate(model, tokenizer)

print(f"Accuracy: {results['accuracy']:.2%}")
print(f"Loss: {results['loss']:.4f}")
```

## Multiple Benchmarks

```python
from gpt_lab.benchmarks import (
    FillInTheBlankBenchmark,
    MultipleChoiceBenchmark
)

benchmarks = {
    "asdiv": FillInTheBlankBenchmark("asdiv"),
    "hellaswag": MultipleChoiceBenchmark("hellaswag"),
    "wiki_qa": MultipleChoiceBenchmark("wiki_qa")
}

all_results = {}
for name, benchmark in benchmarks.items():
    result = benchmark.evaluate(model, tokenizer)
    all_results[name] = result
    print(f"{name}: {result['accuracy']:.2%}")
```

## Creating Custom Benchmarks

```python
# experiments/my_exp/gpt_lab/benchmarks/my_benchmark.py

from gpt_lab.benchmarks import BaseBenchmark

class MyCustomBenchmark(BaseBenchmark):
    """Evaluate model on custom task."""
    
    def __init__(self, dataset_path, num_samples=1000):
        self.dataset_path = dataset_path
        self.num_samples = num_samples
        self.data = self._load_data()
    
    def _load_data(self):
        """Load benchmark data."""
        # Load from dataset_path
        return data
    
    def evaluate(self, model, tokenizer=None):
        """
        Run evaluation.
        
        Args:
            model: Model to evaluate
            tokenizer: Optional tokenizer
        
        Returns:
            dict: Evaluation metrics
        """
        correct = 0
        total = 0
        
        for sample in self.data[:self.num_samples]:
            prediction = model.predict(sample['input'])
            if prediction == sample['target']:
                correct += 1
            total += 1
        
        accuracy = correct / total
        
        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": total
        }
```

## Running Benchmarks

### From Python

```python
from gpt_lab.benchmarks import MyBenchmark

benchmark = MyBenchmark(dataset_path="data/my_dataset.json")
results = benchmark.evaluate(model, tokenizer)
```

### From CLI

```bash
python -m gpt_lab.benchmarks.runner \
    --model path/to/model.pt \
    --benchmarks hellaswag,wiki_qa,asdiv \
    --output results.json
```

## Benchmark Results Format

All benchmarks should return a dictionary with at least:

```python
{
    "accuracy": float,  # Primary metric (0-1)
    "loss": float,      # Optional loss value
    # Additional task-specific metrics
}
```

## Testing

Benchmark functionality is tested as part of the catalog tests:

```bash
pytest src/gpt_lab/benchmarks/tests/ -v
```

## Contributing

Add benchmarks to:
- Pack: `catalogs/packs/<pack>/gpt_lab/benchmarks/`
- Experiment: `experiments/<exp>/gpt_lab/benchmarks/`

Guidelines:
- Subclass `BaseBenchmark` if available
- Implement `evaluate(model, tokenizer)` method
- Return standardized metrics dictionary
- Document dataset requirements
- Provide data loading utilities

Example structure:

```python
class NewBenchmark:
    """
    Evaluate models on [task name].
    
    Dataset: [name and source]
    Metrics: [list of metrics]
    """
    
    def __init__(self, **kwargs):
        # Setup
        pass
    
    def evaluate(self, model, tokenizer=None):
        """Run evaluation."""
        # Implementation
        return {"accuracy": accuracy, ...}
```

## Integration with Models

Models can declare benchmark compatibility:

```python
from gpt_lab.models import MyModel

class MyModel:
    @benchmark_compatible("multiple_choice")
    def answer_mc(self, question, choices):
        # Implementation
        pass
```

This allows benchmarks to automatically discover and use the appropriate model methods.

## Visualization

Use the experiment comparison notebook to analyze benchmark results:

```bash
marimo edit notebooks/experiment_comparison.py
```

Load multiple experiment runs and compare their benchmark scores across different models or configurations.

