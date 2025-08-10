from typing import List, Dict, Optional, Type, Callable, Any, Union, Tuple, Sequence
from dataclasses import dataclass, field
import os
import sys
import importlib
import inspect

import torch
import torch.nn as nn


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

    # Use a backing field that is not part of the constructor
    _source_module_id: str = field(init=False, repr=False, default="")

    @property
    def source_module_id(self) -> str:
        return self._source_module_id

    @source_module_id.setter
    def source_module_id(self, value: str):
        marker = os.path.join("modules", "catalog") + os.sep
        _, _, tail = value.partition(marker)
        source_file = tail if tail else value
        self._source_module_id = os.path.splitext(source_file)[0].replace(os.path.sep, '_')


##################################################
######## TOOLS THE USER MIGHT WANT TO USE ########
##################################################


class SkipModuleException(Exception):
    """A special exception used to signal that a module should be skipped during test discovery."""
    pass

def ignore_test_if_no_cuda():
    if not torch.cuda.is_available():
        raise SkipModuleException("Module requires CUDA, but it is not available.")
    return


##################################################
### TOOLS THE USER DOESN'T HAVE TO WORRY ABOUT ###
##################################################


def list_all_files_in_folder_and_subdirs(folder_path: str) -> List[str]:
    """
    Recursively list all .py files in the given folder and its subdirectories.
    Returns a list of file paths relative to the folder_path.
    """
    all_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if not file.endswith('.py'):
                continue
            rel_dir = os.path.relpath(root, folder_path)
            if rel_dir == ".":
                rel_file = file
            else:
                rel_file = os.path.join(rel_dir, file)
            all_files.append(rel_file)
    return all_files


def get_available_devices(exclude: List[str] = []) -> Tuple[List[str]]:
    available_devices = []
    available_devices_with_ranks = []
    world_size = torch.cuda.device_count()

    # Check for CUDA devices
    if torch.cuda.is_available():
        available_devices.append('cuda')

        if world_size > 1:
            for i in range(world_size):
                available_devices_with_ranks.append(f'cuda:{i}')

    # Check for MPS devices (Apple Silicon)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        available_devices.append('mps')

    # Check for HIP devices (AMD ROCm)
    if hasattr(torch, 'has_hip') and torch.has_hip and torch.device('hip').type == 'hip':
        available_devices.append('hip')

        if world_size > 1:
            for i in range(world_size):
                available_devices_with_ranks.append(f'hip:{i}')

    if exclude:
        def should_keep(dev):
            return not any(ex in dev for ex in exclude)
        available_devices = [dev for dev in available_devices if should_keep(dev)]
        available_devices_with_ranks = [dev for dev in available_devices_with_ranks if should_keep(dev)]

    # Only include CPU if there are no other options
    if not available_devices:
        available_devices.append('cpu')

    return available_devices, available_devices_with_ranks


def discover_dunder_objects(
        dunder: str, 
        object: Any,
        excluded_files: List[str] = [],
        search_folders: Union[None, str, List[str]] = None
    ) -> Tuple[List[Any], Dict[str, Exception]]:
    """
    Discover objects with a given dunder name in Python files within specified folders.

    Args:
        dunder: The dunder attribute name to look for (e.g., '__test_config__').
        object: The type or class to check isinstance(obj, object).
        excluded_files: List of filenames to exclude from search.
        search_folders: A folder path, or list of folder paths, to search. If None, uses the current directory.

    Returns:
        A tuple containing:
        - A list of discovered objects.
        - A dictionary mapping filenames to the exceptions that occurred while processing them.
    """
    import os
    import sys
    import importlib

    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)

    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    # Determine which folders to search
    if search_folders is None:
        folders_to_search = [current_dir]
    elif isinstance(search_folders, str):
        folders_to_search = [os.path.abspath(search_folders)]
    else:
        folders_to_search = [os.path.abspath(f) for f in search_folders]

    all_files = []
    for folder_path in folders_to_search:
        all_files.extend(list_all_files_in_folder_and_subdirs(folder_path))

    all_files = [f for f in all_files if os.path.basename(f) not in excluded_files]

    objects = []
    errors = {}
    for file in all_files:
        try:
            # Figure out the correct module name for importlib
            # file may be absolute or relative to project_root
            abs_file_path = os.path.abspath(os.path.join(folder_path if not os.path.isabs(file) else '', file))
            relative_path = os.path.relpath(abs_file_path, project_root)
            module_name = relative_path.replace('.py', '').replace(os.sep, '.')
            
            module = importlib.import_module(module_name)

            obj = getattr(module, dunder, None)
            if isinstance(obj, BenchmarkConfig):
                obj.source_module_id = abs_file_path
            if isinstance(obj, object):
                objects.append(obj)
        except SkipModuleException:
            # This is a graceful skip, not an error.
            continue
        except Exception as e:
            errors[file] = e

    return objects, errors


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
