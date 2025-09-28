from abc import ABC, abstractmethod
from typing import Callable, Iterable, Any, Dict, List
import inspect

from tqdm import tqdm


def register_handler(benchmark_type: str) -> Callable:
    """
    A decorator to register a model's method as a handler for a specific benchmark type.
    This works by attaching a `_benchmark_type` attribute to the decorated method.

    Args:
        benchmark_type: The name of the benchmark type (e.g., "multiple_choice").

    Returns:
        A decorator that registers the function.
    """
    def decorator(fn: Callable) -> Callable:
        fn._benchmark_type = benchmark_type
        return fn
    return decorator


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
        self.handler = self._find_handler(model, benchmark_type)
        if self.handler is None:
            available_handlers = self._find_all_handlers(model)
            raise AttributeError(
                f"No handler registered for benchmark type '{benchmark_type}' on model '{model.__class__.__name__}'. "
                f"Available handlers on this model: {available_handlers}"
            )
        self.results = self._initialize_metrics()

    def _find_handler(self, model: Any, benchmark_type: str) -> Callable:
        """Finds a benchmark handler method on the model instance."""
        for _, method in inspect.getmembers(model, predicate=inspect.ismethod):
            if getattr(method, '_benchmark_type', None) == benchmark_type:
                return method
        return None

    def _find_all_handlers(self, model: Any) -> List[str]:
        """Finds all available benchmark handlers on the model instance for better error messages."""
        handlers = []
        for _, method in inspect.getmembers(model, predicate=inspect.ismethod):
            if hasattr(method, '_benchmark_type'):
                handlers.append(getattr(method, '_benchmark_type'))
        return handlers

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
                # The handler is now a bound method, so we don't pass the model instance.
                model_outputs = self.handler(batch)
                self._process_results_batch(batch, model_outputs)
                batch = []

        # Process any remaining items in the last batch
        if batch:
            model_outputs = self.handler(batch)
            self._process_results_batch(batch, model_outputs)

        return self._compute_final_metrics()
