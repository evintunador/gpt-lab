from typing import Union, Tuple, Any

import torch
import torch.nn as nn

from nn_modules.catalog_utils import (
    ModuleTestConfig, 
    BenchmarkConfig, 
    Competitor,
    next_multiple,
)
from nn_modules.catalog.activations import ReLU2
from .fp8_linear import FP8Linear, is_hopper_available
from .mlp import (
    mlp_input_args, 
    mlp_tolerances, 
    mlp_dims_to_test, 
    mlp_dtypes_to_test, 
    mlp_activations_to_test,
    output_validator,
    mlp_dims_to_bench,
    mlp_activations_to_bench,
    mlp_dtypes_to_bench,
    benchmark_input_provider,
)


torch.set_float32_matmul_precision('medium')
torch._dynamo.config.recompile_limit = 100


##################################################
############# PRIMARY PYTORCH MODULE #############
##################################################


class GatedMLP(nn.Module):
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
        self.Wgate = FP8Linear(in_features=self.in_dim, out_features=self.hidden_dim, fp8=fp8)
        self.Wdown = FP8Linear(in_features=self.hidden_dim, out_features=self.out_dim, fp8=fp8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.Wdown(self.Wup(x) * self.act_fn(self.Wgate(x)))


########################################################
# PRECOMPILED IMPLEMENTATION FOR TESTING torch.compile #
########################################################

@torch.compile
def fwd(inp, w_up, w_gate, w_down, act_fn):
    return w_down(w_up(inp) * act_fn(w_gate(inp)))

class PreCompiledGatedMLP(GatedMLP):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return fwd(x, self.Wup, self.Wgate, self.Wdown, self.act_fn)
    

def pre_compiled_run_filter(inputs: Union[torch.Tensor, Tuple[Any]]) -> bool:
    """
    Many custom modules are only appropriate for use under a subset of all the conditions where a regular pytorch nn.module can run.
    Use this function to ensure that testing is only attempted on that subset.
    Here, for example, our PreCompiledGatedMLP should only be run on a GPU since it uses torch.compile.
    """
    if 'cpu' in str(inputs[0].device):
        return False
    return True


##################################################
#################### TESTING ####################
##################################################


__competitors__ = {
    'GatedMLP': Competitor(module_class=GatedMLP),
    'PreCompiledGatedMLP': Competitor(module_class=PreCompiledGatedMLP, run_filter=pre_compiled_run_filter),
}


__test_config__ = ModuleTestConfig(
    competitors=__competitors__,
    reference_competitor='GatedMLP',
    test_cases=[
        {
            'init_args': {'in_dim': dim, 'out_dim': dim, 'hidden_dim': int((dim * 4) * (2/3)), 'activation': act, 'dtype': dt, 'fp8': fp8},
            'input_args': lambda dev, dim=dim, dt=dt: mlp_input_args(device=dev, dim=dim, dtype=dt),
            'output_validator': output_validator,
            'tolerances': mlp_tolerances(dt), # Optional
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


__benchmark_config__ = BenchmarkConfig(
    module_name='GatedMLP',
    competitors=__competitors__,
    parameter_space={
        'dim': mlp_dims_to_bench,
        'hidden_mult': [2, 4, 8],
        'activation': mlp_activations_to_bench,
        'dtype': mlp_dtypes_to_bench,
        'fp8': ([True, False] if is_hopper_available() else [False])
    },
    init_arg_builder=lambda params: {
        'in_dim': params['dim'],
        'out_dim': params['dim'],
        'hidden_dim': next_multiple(params['dim'] * params['hidden_mult'] / 3, n=128),
        'activation': params['activation'],
        'dtype': params['dtype'],
        'fp8': params['fp8']
    },
    input_provider=benchmark_input_provider,
)