from typing import Union, Tuple, Any

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
from .mlp import (
    input_args, 
    tolerances_fn, 
    dims_to_test, 
    act_to_test,
    output_validator,
    dims_to_bench,
    hidden_mult_to_bench,
    benchmark_input_provider,
)


torch.set_float32_matmul_precision('medium')
torch._dynamo.config.recompile_limit = 100


##################################################
############# PRIMARY PYTORCH MODULE #############
##################################################


class GLU(nn.Module):
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
        hidden_dim = hidden_dim if hidden_dim is not None else int(in_dim * 8/3)
        self.hidden_dim = next_multiple(x=hidden_dim, n=128)

        self.act_str = activation.lower()
        act_registry = {
            "relu": nn.ReLU(),
            "silu": nn.SiLU(),
            "relu2": ReLU2(),
        }
        self.act_fn = act_registry[self.act_str]

        self.Wup_gate = FP8Linear(in_features=self.in_dim, out_features=self.hidden_dim * 2, fp8=fp8)
        self.Wdown = FP8Linear(in_features=self.hidden_dim, out_features=self.out_dim, fp8=fp8)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        up_gate = self.Wup_gate(x)
        up, gate = up_gate.chunk(2, dim=-1)
        return self.dropout(self.Wdown(up * self.act_fn(gate)))


########################################################
# PRECOMPILED IMPLEMENTATION FOR TESTING torch.compile #
########################################################

@torch.compile(mode='default')
def pre_compiled_fwd(inp, w_up_gate, w_down, act_fn, dropout):
    up_gate = w_up_gate(inp)
    up, gate = up_gate.chunk(2, dim=-1)
    return dropout(w_down(up * act_fn(gate)))

class PreCompiledGLU(GLU):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return pre_compiled_fwd(x, self.Wup_gate, self.Wdown, self.act_fn, self.dropout)
    

def pre_compiled_run_filter(inputs: Union[torch.Tensor, Tuple[Any]]) -> bool:
    """
    Many custom modules are only appropriate for use under a subset of all the conditions where a regular pytorch nn.module can run.
    Use this function to ensure that testing is only attempted on that subset.
    Here, for example, our PreCompiledGLU should only be run on a GPU since it uses torch.compile.
    """
    if 'cpu' in str(inputs[0].device):
        return False
    return True


##################################################
#################### TESTING ####################
##################################################


__competitors__ = {
    'GLU': Competitor(module_class=GLU),
    'PreCompiledGLU': Competitor(module_class=PreCompiledGLU, run_filter=pre_compiled_run_filter),
}


__test_config__ = ModuleTestConfig(
    competitors=__competitors__,
    reference_competitor='GLU',
    test_cases=[
        {
            'init_args': {'in_dim': dim, 'activation': act, 'fp8': fp8},
            'input_args': input_args(dim=dim), 
            'output_validator': output_validator,
            'tolerances_fn': tolerances_fn,                  # Optional
            'case_descriptor': f'dim={dim}_act={act}_fp8={fp8}',
        }
        for dim in dims_to_test
        for act in act_to_test
        for fp8 in ([True, False] if is_hopper_available() else [False])
    ]
)


##################################################
################# BENCHMARKING ###################
##################################################


__benchmark_config__ = BenchmarkConfig(
    module_name='GLU',
    competitors=__competitors__,
    parameter_space={
        'dim': dims_to_bench,
        'hidden_mult': hidden_mult_to_bench,
        'activation': act_to_test,
        'fp8': ([True, False] if is_hopper_available() else [False]),
    },
    init_arg_builder=lambda params: {
        'in_dim': params['dim'],
        'hidden_dim': int(params['dim'] * params['hidden_mult'] * 2/3),
        'activation': params['activation'],
        'fp8': params['fp8']
    },
    input_provider=benchmark_input_provider,
)