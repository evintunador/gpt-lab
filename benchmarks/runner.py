from abc import ABC, abstractmethod
from typing import Iterable, Any, Dict, List

from tqdm import tqdm

from .registry import get_handler


class BenchmarkRunner(ABC):
    """
    An abstract base class for running benchmarks.

    This class provides the generic structure for iterating over a dataset,
    passing data to the appropriate model handler, and processing the results.
    Subclasses must implement the logic for initializing metrics, processing
    a batch of results, and computing the final metrics.
    """
    def __init__(self, model: Any, benchmark_type: str):
        self.model = model
        self.benchmark_type = benchmark_type
        self.handler = get_handler(benchmark_type)
        self.results = self._initialize_metrics()

    @abstractmethod
    def _initialize_metrics(self) -> Dict[str, Any]:
        """Initializes a dictionary to store metric-related data."""
        pass

    @abstractmethod
    def _process_results_batch(
        self,
        batch: List[Any],
        model_outputs: Any
    ) -> None:
        """
        Processes the model's outputs for a batch and updates metrics.
        
        Args:
            batch: The list of raw data items from the dataset.
            model_outputs: The output from the model's registered handler.
        """
        pass

    @abstractmethod
    def _compute_final_metrics(self) -> Dict[str, Any]:
        """Computes and returns the final metrics dictionary."""
        pass

    def run(
        self,
        dataset: Iterable[Any],
        batch_size: int = 1,
        limit: int = None
    ) -> Dict[str, Any]:
        """
        Runs the benchmark.

        Args:
            dataset: An iterable dataset yielding data items.
            batch_size: The number of items to process in a batch.
            limit: The maximum number of items to process from the dataset.

        Returns:
            A dictionary containing the final computed metrics.
        """
        batch = []
        
        # We wrap the dataset with tqdm for a progress bar
        dataset_iterator = (
            tqdm(dataset, total=limit) if limit is not None else tqdm(dataset)
        )
        
        for i, item in enumerate(dataset_iterator):
            if limit and i >= limit:
                break
            
            batch.append(item)
            
            if len(batch) == batch_size:
                # Pass the batch to the model's registered handler
                model_outputs = self.handler(self.model, batch)
                self._process_results_batch(batch, model_outputs)
                batch = []

        # Process any remaining items in the last batch
        if batch:
            model_outputs = self.handler(self.model, batch)
            self._process_results_batch(batch, model_outputs)

        return self._compute_final_metrics()
