from dataclasses import dataclass
from typing import Any, Dict, List

import torch

from gpt_lab.benchmarks.runner import BenchmarkRunner
from gpt_lab.benchmarks.stats_utils import calculate_bootstrap_ci


@dataclass
class MultipleChoiceItem:
    """
    Standardized data format for a single item in a multiple choice benchmark.
    """
    context: str
    choices: List[str]
    label: int


class MultipleChoiceBenchmark(BenchmarkRunner):
    """
    A concrete benchmark runner for multiple-choice tasks.

    It calculates accuracy by comparing model predictions to labels.
    The model handler for this benchmark is expected to return a list of
    integer predictions, one for each item in the batch.
    """
    def __init__(self, model: Any):
        super().__init__(model, benchmark_type="multiple_choice")

    def _initialize_metrics(self) -> Dict[str, Any]:
        return {"correct": 0, "total": 0, "results_list": []}

    def _process_results_batch(
        self,
        batch: List[MultipleChoiceItem],
        predictions: List[int]
    ) -> None:
        """
        Compares predictions with labels and updates the counts.
        """
        for item, pred in zip(batch, predictions):
            is_correct = 1 if pred == item.label else 0
            self.results["results_list"].append(is_correct)
            if is_correct:
                self.results["correct"] += 1
            self.results["total"] += 1

    def _compute_final_metrics(self) -> Dict[str, Any]:
        """
        Calculates the final accuracy.
        """
        total = self.results["total"]
        if total == 0:
            return {"accuracy": 0.0, "accuracy_ci": (0.0, 0.0), "total_examples": 0}

        accuracy = self.results["correct"] / total
        accuracy_ci = calculate_bootstrap_ci(self.results["results_list"])
        
        return {
            "accuracy": accuracy, 
            "accuracy_ci": accuracy_ci,
            "total_examples": total
        }

    @staticmethod
    def render_example(example: MultipleChoiceItem, tokenizer_encode_fn):
        """
        Given the example as a MultipleChoiceItem, render it as three torch tensors:
        - tokens (the tokens of context + completion, of size KxN, where K is the number of candidates)
        - mask (is 1 in the region of the candidate completion, where we evaluate likelihoods)
        - label (the index of the correct completion, which we hope has the highest likelihood)
        """
        ctx = example.context
        label = example.label
        endings = example.choices
        num_choices = len(endings)

        # gather up all the tokens
        ctx_tokens = tokenizer_encode_fn(ctx)
        tok_rows = []
        mask_rows = []
        for end in endings:
            end_tokens = tokenizer_encode_fn(" " + end)  # NOTE: prepending " " assuming GPT-2 based tokenizer
            tok_rows.append(ctx_tokens + end_tokens)
            mask_rows.append([0]*len(ctx_tokens) + [1]*len(end_tokens))

        # have to be careful during the collation because the number of tokens in each row can differ
        max_len = max(len(row) for row in tok_rows)
        tokens = torch.zeros((num_choices, max_len), dtype=torch.long)
        mask = torch.zeros((num_choices, max_len), dtype=torch.long)
        for i, (tok_row, mask_row) in enumerate(zip(tok_rows, mask_rows)):
            tokens[i, :len(tok_row)] = torch.tensor(tok_row)
            mask[i, :len(mask_row)] = torch.tensor(mask_row)

        return tokens, mask, label

"""
Example of how the class above might be used in a script:

from some_model_file import MyModel
from gpt_lab.data_sources.catalog.benchmarks.multiple_choice import HellaSwagDataset

1. Instantiate the model and dataset
model = MyModel()
dataset = HellaSwagDataset(split="val")

2. Instantiate the benchmark runner
benchmark = MultipleChoiceBenchmark(model)

3. Run the evaluation
results = benchmark.run(dataset, batch_size=8, limit=1000)

print(results)
Expected output: {'accuracy': 0.45, 'total_examples': 1000} (example value)
"""