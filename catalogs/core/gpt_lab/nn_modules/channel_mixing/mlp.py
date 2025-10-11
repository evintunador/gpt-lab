from typing import List, Union, Tuple, Any
import math

import torch
import torch.nn as nn

from gpt_lab.nn_modules.catalog_utils import (
    ModuleTestConfig, 
    BenchmarkConfig, 
    Competitor,
    next_multiple,
)
from gpt_lab.nn_modules.activations.relu2 import ReLU2
from .fp8_linear import FP8Linear, is_hopper_available


torch.set_float32_matmul_precision('medium')
torch._dynamo.config.recompile_limit = 100


##################################################
############# PRIMARY PYTORCH MODULE #############
##################################################


class MLP(nn.Module):
    def __init__(
        self, 
        in_dim: int, 
        hidden_dim: int | None = None, 
        out_dim: int | None = None, 
        activation: str = "silu", 
        dropout: float = 0.0, 
        fp8: bool = False
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim if out_dim is not None else in_dim
        hidden_dim = hidden_dim if hidden_dim is not None else in_dim * 4
        self.hidden_dim = next_multiple(x=hidden_dim, n=128)

        self.act_str = activation.lower()
        act_registry = {
            "relu": nn.ReLU(),
            "silu": nn.SiLU(),
            "relu2": ReLU2(),
            "gelu": nn.GELU(),
        }
        self.act_fn = act_registry[self.act_str]

        self.Wup = FP8Linear(in_features=self.in_dim, out_features=self.hidden_dim, fp8=fp8) 
        self.Wdown = FP8Linear(in_features=self.hidden_dim, out_features=self.out_dim, fp8=fp8)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.Wdown(self.act_fn(self.Wup(x))))


########################################################
# PRECOMPILED IMPLEMENTATION FOR TESTING torch.compile #
########################################################

@torch.compile
def fwd(inp, w_up, w_down, act_fn, dropout):
    return dropout(w_down(act_fn(w_up(inp))))

class PreCompiledMLP(MLP):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return fwd(x, self.Wup, self.Wdown, self.act_fn, self.dropout)
    

def pre_compiled_run_filter(inputs: Union[torch.Tensor, Tuple[Any]]) -> bool:
    """
    Many custom modules are only appropriate for use under a subset of all the conditions where a regular pytorch nn.module can run.
    Use this function to ensure that testing is only attempted on that subset.
    Here, for example, our PreCompiledMLP should only be run on a GPU since it uses torch.compile.
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
    assert output_tensor.device == output_tensor.device


__competitors__ = {
    'MLP': Competitor(module_class=MLP),
    'PreCompiledMLP': Competitor(module_class=PreCompiledMLP, run_filter=pre_compiled_run_filter),
}


def input_args(device: str, dim: int, dtype):
    return (torch.randn(128, dim, device=device, dtype=dtype, requires_grad=True),)
def tolerances(dtype: torch.dtype) -> dict:
    if dtype == torch.float32:
        return {'atol': 1e-2, 'rtol': 1e-1}
    elif dtype == torch.float16:
        return {'atol': 5e-2, 'rtol': 1}
    elif dtype == torch.bfloat16:
        return {'atol': 1e-1, 'rtol': 10}
    else:
        return {'atol': 1e-2, 'rtol': 1e100000}
dims_to_test = [768]
dtypes_to_test = [torch.float32, torch.float16, torch.bfloat16]
act_to_test = ['relu', 'relu2', 'silu']


__test_config__ = ModuleTestConfig(
    competitors=__competitors__,
    reference_competitor='MLP',
    test_cases=[
        {
            'init_args': {'in_dim': dim, 'activation': act, 'fp8': fp8},
            'input_args': lambda dev, dim=dim, dt=dt: input_args(device=dev, dim=dim, dtype=dt),
            'output_validator': output_validator,
            'tolerances': tolerances(dt),          # Optional
            'case_descriptor': f'dim={dim}_dt={dt}_act={act}_fp8={fp8}',
        }
        for dim in dims_to_test
        for dt in dtypes_to_test
        for act in act_to_test
        for fp8 in ([True, False] if is_hopper_available() else [False])
    ]
)


##################################################
################# BENCHMARKING ###################
##################################################


dims_to_bench = [128, 512, 2048]
hidden_mult_to_bench = [2, 4, 8]


def benchmark_input_provider(init_args: dict, device: str) -> tuple:
    """Generates a standard input for benchmarking."""
    # input shape: (batch_size, sequence_length, dimension)
    dtype = init_args.get('dtype', torch.float32)
    return (torch.randn(512, init_args['in_dim'], device=device, dtype=dtype),)


__benchmark_config__ = BenchmarkConfig(
    module_name='MLP',
    competitors=__competitors__,
    parameter_space={
        'dim': dims_to_bench,
        'hidden_mult': hidden_mult_to_bench,
        'activation': act_to_test,
        'fp8': [True, False] if is_hopper_available() else [False],
    },
    init_arg_builder=lambda params: {
        'in_dim': params['dim'],
        'hidden_dim': params['dim'] * params['hidden_mult'],
        'activation': params['activation'],
        'fp8': params['fp8']
    },
    input_provider=benchmark_input_provider,
)