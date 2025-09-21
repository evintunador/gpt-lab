from abc import ABC, abstractmethod
from typing import Callable, Iterable, Any, Dict, List

from tqdm import tqdm


# The registry is a simple dictionary mapping benchmark type names to handler functions.
# The handler function is expected to be a method on a model class.
BENCHMARK_HANDLERS: Dict[str, Callable] = {}


def register_handler(benchmark_type: str) -> Callable:
    """
    A decorator to register a model's method as a handler for a specific benchmark type.

    Args:
        benchmark_type: The name of the benchmark type (e.g., "logit_multiple_choice").

    Returns:
        A decorator that registers the function.
    """
    def decorator(fn: Callable) -> Callable:
        # We store the function itself. The runner will be responsible for passing
        # the model instance (`self`) as the first argument.
        BENCHMARK_HANDLERS[benchmark_type] = fn
        return fn
    return decorator


def get_handler(benchmark_type: str) -> Callable:
    """
    Retrieves the handler for a given benchmark type from the registry.

    Args:
        benchmark_type: The name of the benchmark type.

    Returns:
        The registered handler function.

    Raises:
        KeyError: If no handler is registered for the given benchmark type.
    """
    try:
        return BENCHMARK_HANDLERS[benchmark_type]
    except KeyError:
        raise KeyError(
            f"No handler registered for benchmark type '{benchmark_type}'. "
            f"Available handlers: {list(BENCHMARK_HANDLERS.keys())}"
        )


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
