import numpy as np
from typing import List, Callable, Tuple


def calculate_bootstrap_ci(
    data: List[float],
    statistic_fn: Callable[[np.ndarray], float] = np.mean,
    n_resamples: int = 1000,
    confidence_level: float = 0.95
) -> Tuple[float, float]:
    """
    Calculates the bootstrap confidence interval for a given statistic.

    Args:
        data: A list or array of numerical data.
        statistic_fn: The function to compute the statistic (e.g., np.mean, np.median).
        n_resamples: The number of bootstrap samples to generate.
        confidence_level: The desired confidence level (e.g., 0.95 for 95% CI).

    Returns:
        A tuple containing the lower and upper bounds of the confidence interval.
    """
    if len(data) == 0:
        return (0.0, 0.0)

    data = np.array(data)
    bootstrap_samples = np.random.choice(data, (n_resamples, len(data)), replace=True)
    bootstrap_statistics = np.array([statistic_fn(sample) for sample in bootstrap_samples])

    lower_percentile = (1.0 - confidence_level) / 2.0 * 100
    upper_percentile = (100 - lower_percentile)
    
    lower_bound = np.percentile(bootstrap_statistics, lower_percentile)
    upper_bound = np.percentile(bootstrap_statistics, upper_percentile)

    return (float(lower_bound), float(upper_bound))
