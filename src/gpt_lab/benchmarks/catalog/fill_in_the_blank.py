from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
import numpy as np

from gpt_lab.benchmarks.runner import BenchmarkRunner
from gpt_lab.benchmarks.stats_utils import calculate_bootstrap_ci


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
        return {
            "exact_match": 0, 
            "total_nll": 0.0, 
            "total": 0,
            "exact_match_list": [],
            "nll_list": [],
        }

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
            is_correct = 1 if pred_str.strip() == item.answer.strip() else 0
            self.results["exact_match_list"].append(is_correct)
            self.results["nll_list"].append(nll)
            
            if is_correct:
                self.results["exact_match"] += 1
            self.results["total_nll"] += nll
            self.results["total"] += 1

    def _compute_final_metrics(self) -> Dict[str, Any]:
        total = self.results["total"]
        if total == 0:
            return {
                "exact_match_accuracy": 0.0,
                "exact_match_accuracy_ci": (0.0, 0.0),
                "average_nll": 0.0,
                "perplexity": float('inf'),
                "perplexity_ci": (float('inf'), float('inf')),
                "total_examples": 0
            }

        exact_match_acc = self.results["exact_match"] / total
        exact_match_acc_ci = calculate_bootstrap_ci(self.results["exact_match_list"])
        
        avg_nll = self.results["total_nll"] / total
        perplexity = np.exp(avg_nll)
        
        nll_ci = calculate_bootstrap_ci(self.results["nll_list"])
        perplexity_ci = (np.exp(nll_ci[0]), np.exp(nll_ci[1]))

        return {
            "exact_match_accuracy": exact_match_acc,
            "exact_match_accuracy_ci": exact_match_acc_ci,
            "average_nll": avg_nll,
            "perplexity": perplexity,
            "perplexity_ci": perplexity_ci,
            "total_examples": total,
        }
