from typing import Callable, Dict, Any

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
