from typing import List, Dict, Optional, Type, Callable, Any, Union, Tuple, Sequence
from dataclasses import dataclass

import torch
import torch.nn as nn

from catalog_utils import SkipModuleException


##################################################
#### INHERIT FROM THESE FOR TESTS/BENCHMARKS #####
##################################################


@dataclass
class TensorParallelConfig:
    """Configuration for tensor parallelism."""
    parallelize_plan: Dict[str, Any]
    tp_mesh_adjust_fn: Optional[Callable[[nn.Module, Any], nn.Module]] = None


@dataclass
class Competitor:
    """A competitor for testing or benchmarking."""
    module_class: Type[nn.Module]
    tp_config: Optional[TensorParallelConfig] = None
    run_filter: Callable[Union[torch.Tensor, Tuple[Any]], bool] = None


@dataclass
class ModuleTestConfig:
    """
    A dataclass to hold the full testing configuration for a module.
    """
    # A dictionary mapping competitor names to their configurations.
    competitors: Dict[str, Competitor]
    # The name of the competitor to be used as the reference for correctness checks.
    reference_competitor: str
    # A list of dictionaries, where each dict is a self-contained test case.
    test_cases: List[Dict]


@dataclass
class BenchmarkConfig:
    """
    Holds the complete benchmarking configuration for a module.
    This defines the matrix of parameters to sweep over.
    """
    # A friendly name for the module, used for filenaming.
    module_name: str
    # A dictionary mapping competitor names to their configurations.
    competitors: Dict[str, Competitor]
    # The parameter space to sweep.
    # e.g., {'dim': [1024, 2048], 'activation': ['relu', 'silu'], 'dtype': [torch.float16]}
    parameter_space: Dict[str, List[Any]]
    # A function that takes a dictionary of a single parameter combination
    # from the sweep and returns the full init_args for the module.
    # This is useful for args that are derived from others (e.g., out_dim=dim)
    init_arg_builder: Callable[[Dict[str, Any]], Dict[str, Any]]
    # A function that provides the input tensors for the module, given the init_args.
    input_provider: Callable[[Dict[str, Any], str], tuple]

    def __post_init__(self):
        self.module_name = self.module_name.replace('/', '_').replace('\\', '_')

##################################################
######## TOOLS THE USER MIGHT WANT TO USE ########
##################################################


def ignore_test_if_no_cuda():
    if not torch.cuda.is_available():
        raise SkipModuleException("Module requires CUDA, but it is not available.")
    return


def next_multiple(x, n):
    return int(((int(x) + n - 1) // n) * n)


##################################################
### TOOLS THE USER DOESN'T HAVE TO WORRY ABOUT ###
##################################################


def get_total_loss(outputs: Union[torch.Tensor, Sequence[torch.Tensor]]) -> torch.Tensor:
    """Computes a scalar loss from a single tensor or a tuple of tensors."""
    if isinstance(outputs, torch.Tensor):
        # Handles the common case of a single tensor output
        return outputs.sum()
    
    # Handles tuple outputs, summing only the floating point tensors
    total_loss = torch.tensor(0.0).to(outputs[0].device)
    for out in outputs:
        if isinstance(out, torch.Tensor) and out.is_floating_point():
            total_loss += out.sum()
    return total_loss
