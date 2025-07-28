from typing import List, Union, Tuple, Any
import math

import torch
import torch.nn as nn

from modules.base_test_bench_utils import (
    ModuleTestConfig, 
    BenchmarkConfig, 
    Competitor,
    TensorParallelConfig,
)
from modules.catalog.utils import next_multiple
from modules.catalog.activations.relu2 import ReLU2
from modules.catalog.fp8_linear import FP8Linear, is_hopper_available
from modules.catalog.channel_mixing.mlp import (
    mlp_input_args, mlp_tolerances, mlp_dims_to_test, mlp_dtypes_to_test, mlp_activations_to_test,
)


##################################################
############# PRIMARY PYTORCH MODULE #############
##################################################


class FP8MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int, activation: str, 
                 dtype: torch.dtype = torch.float32, device: str = 'cpu', fp8 = False):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = next_multiple(x=hidden_dim, n=128)

        self.act_str = activation.lower()
        act_registry = {
            "relu": nn.ReLU(),
            "silu": nn.SiLU(),
            "relu2": ReLU2(),
        }
        self.act_fn = act_registry[self.act_str]

        self.Wup = FP8Linear(in_features=self.in_dim, out_features=self.hidden_dim, fp8=fp8) 
        self.Wdown = FP8Linear(in_features=self.hidden_dim, out_features=self.out_dim, fp8=fp8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.Wdown(self.act_fn(self.Wup(x)))


########################################################
# PRECOMPILED IMPLEMENTATION FOR TESTING torch.compile #
########################################################

@torch.compile
def fwd(inp, w_up, w_down, act_fn):
    return w_down(act_fn(w_up(inp)))

class PreCompiledFP8MLP(FP8MLP):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return fwd(x, self.Wup, self.Wdown, self.act_fn)
    

def pre_compiled_run_filter(inputs: Union[torch.Tensor, Tuple[Any]]) -> bool:
    """
    Many custom modules are only appropriate for use under a subset of all the conditions where a regular pytorch nn.module can run.
    Use this function to ensure that testing is only attempted on that subset.
    Here, for example, our PreCompiledFP8MLP should only be run on a GPU since it uses torch.compile.
    """
    if 'cpu' in str(inputs[0].device):
        return False
    return True


##################################################
#################### TESTING ####################
##################################################


def output_validator(
        module: nn.Module,
        inputs: Tuple[Any],
        outputs: Tuple[Any],
) -> None:
    """
    Validates whether the base module output meets expectations.
    Testing framework always passes in tuples even if there's only one input/output tensor
    """
    input_tensor = inputs[0] 
    output_tensor = outputs[0]
    expected_shape = (*input_tensor.shape[:-1], module.out_dim)
    assert output_tensor.shape == expected_shape, f"Expected output shape {expected_shape}, but got {output_tensor.shape}"
    assert output_tensor.dtype == input_tensor.dtype
    

__competitors__ = {
    'FP8MLP': Competitor(module_class=FP8MLP),
    'PreCompiledFP8MLP': Competitor(module_class=PreCompiledFP8MLP, run_filter=pre_compiled_run_filter),
}


__test_config__ = ModuleTestConfig(
    competitors=__competitors__,
    reference_competitor='FP8MLP',
    test_cases=[
        {
            'init_args': {'in_dim': dim, 'out_dim': dim, 'hidden_dim': dim * 4, 'activation': act, 'dtype': dt, 'fp8': fp8},
            'input_args': lambda dev, dim=dim, dt=dt: mlp_input_args(device=dev, dim=dim, dtype=dt),
            'output_validator': output_validator,
            'tolerances': mlp_tolerances(dt),          # Optional
            'case_descriptor': f'dim={dim}_dt={dt}_act={act}_fp8={fp8}',
        }
        for dim in mlp_dims_to_test
        for dt in mlp_dtypes_to_test
        for act in mlp_activations_to_test
        for fp8 in ([True, False] if is_hopper_available() else [False])
    ]
)


##################################################
################# BENCHMARKING ###################
##################################################


def benchmark_input_provider(init_args: dict, device: str) -> tuple:
    """Generates a standard input for benchmarking."""
    # input shape: (batch_size, sequence_length, dimension)
    dtype = init_args.get('dtype', torch.float32)
    return (torch.randn(1, 1, init_args['in_dim'], device=device, dtype=dtype),)

__benchmark_config__ = BenchmarkConfig(
    module_name='FP8MLP',
    competitors=__competitors__,
    parameter_space={
        'dim': [32, 64, 128, 512, 1024, 2048, 4096],
        'activation': ['relu', 'silu', 'relu2'],
        'dtype': [torch.float16, torch.bfloat16, torch.float32],
        'fp8': [True, False] if is_hopper_available() else [False],
    },
    init_arg_builder=lambda params: {
        'in_dim': params['dim'],
        'out_dim': params['dim'],
        'hidden_dim': params['dim'] * 4,
        'activation': params['activation'],
        'dtype': params['dtype'],
        'fp8': params['fp8']
    },
    input_provider=benchmark_input_provider,
)