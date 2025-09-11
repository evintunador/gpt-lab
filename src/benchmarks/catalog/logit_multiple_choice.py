from typing import Any, Dict, List
import torch

from src.benchmarks.protocols import LogitMultipleChoiceItem
from src.benchmarks.runner import BenchmarkRunner


class LogitMultipleChoiceBenchmark(BenchmarkRunner):
    """
    A concrete benchmark runner for logit-based multiple-choice tasks.

    It calculates accuracy by comparing model predictions to labels.
    The model handler for this benchmark is expected to return a list of
    integer predictions, one for each item in the batch.
    """
    def __init__(self, model: Any):
        super().__init__(model, benchmark_type="logit_multiple_choice")

    def _initialize_metrics(self) -> Dict[str, Any]:
        return {"correct": 0, "total": 0}

    def _process_results_batch(
        self,
        batch: List[LogitMultipleChoiceItem],
        predictions: List[int]
    ) -> None:
        """
        Compares predictions with labels and updates the counts.
        """
        for item, pred in zip(batch, predictions):
            if pred == item.label:
                self.results["correct"] += 1
            self.results["total"] += 1

    def _compute_final_metrics(self) -> Dict[str, Any]:
        """
        Calculates the final accuracy.
        """
        total = self.results["total"]
        if total == 0:
            return {"accuracy": 0.0, "total_examples": 0}

        accuracy = self.results["correct"] / total
        return {"accuracy": accuracy, "total_examples": total}


# Example of how this might be used in a script:
#
# from some_model_file import MyFlexModel
# from src.data_sources.catalog.benchmarks.logit_multiple_choice.hellaswag import HellaSwagDataset
#
# # 1. Instantiate the model and dataset
# model = MyFlexModel()
# dataset = HellaSwagDataset(split="val")
#
# # 2. Instantiate the benchmark runner
# benchmark = LogitMultipleChoiceBenchmark(model)
#
# # 3. Run the evaluation
# results = benchmark.run(dataset, batch_size=8, limit=1000)
#
# print(results)
# # Expected output: {'accuracy': 0.45, 'total_examples': 1000} (example value)
