from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
import numpy as np

from benchmarks.runner import BenchmarkRunner


@dataclass
class FillInTheBlankItem:
    """
    Standardized data format for a fill-in-the-blank item.
    """
    prompt: str
    answer: str


class FillInTheBlankBenchmark(BenchmarkRunner):
    """
    A benchmark runner for fill-in-the-blank tasks.
    It calculates Exact Match and Perplexity over the answer sequence.
    """
    def __init__(self, model: Any):
        super().__init__(model, benchmark_type="fill_in_the_blank")

    def _initialize_metrics(self) -> Dict[str, Any]:
        return {"exact_match": 0, "total_nll": 0.0, "total": 0}

    def _process_results_batch(
        self,
        batch: List[FillInTheBlankItem],
        model_outputs: List[Tuple[str, float]]
    ) -> None:
        """
        Processes model outputs and updates metrics.
        model_outputs is a list of (predicted_string, nll_of_true_answer_sequence).
        """
        for item, (pred_str, nll) in zip(batch, model_outputs):
            if pred_str.strip() == item.answer.strip():
                self.results["exact_match"] += 1
            self.results["total_nll"] += nll
            self.results["total"] += 1

    def _compute_final_metrics(self) -> Dict[str, Any]:
        total = self.results["total"]
        if total == 0:
            return {
                "exact_match_accuracy": 0.0,
                "average_nll": 0.0,
                "perplexity": float('inf'),
                "total_examples": 0
            }

        exact_match_acc = self.results["exact_match"] / total
        avg_nll = self.results["total_nll"] / total
        perplexity = np.exp(avg_nll)

        return {
            "exact_match_accuracy": exact_match_acc,
            "average_nll": avg_nll,
            "perplexity": perplexity,
            "total_examples": total,
        }
