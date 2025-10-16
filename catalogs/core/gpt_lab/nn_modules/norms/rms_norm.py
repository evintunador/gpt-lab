from typing import Tuple, Any

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from gpt_lab.nn_modules.catalog_utils import ModuleTestConfig, BenchmarkConfig, Competitor


class RMSNorm(nn.Module):
    def __init__(self, dim: int = None):
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor):
        return F.rms_norm(x, (x.size(-1),))


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
    assert output_tensor.shape == input_tensor.shape, f"Expected output shape {input_tensor.shape}, but got {output_tensor.shape}"
    assert output_tensor.dtype == input_tensor.dtype
    assert output_tensor.device == input_tensor.device


__competitors__ = {
    'RMSNorm': Competitor(module_class=RMSNorm),
}


dims_to_test = [128, 768, 2048]


def input_args(dim: int):
    # Test with various shapes: 2D (batch, dim), 3D (batch, seq, dim)
    return (torch.randn(8, 32, dim, requires_grad=True),)


def tolerances_fn(input_args: Tuple[Any]) -> dict:
    x = input_args[0]
    if x.dtype == torch.float32:
        return {'atol': 1e-5, 'rtol': 1e-4}
    elif x.dtype == torch.float16:
        return {'atol': 1e-3, 'rtol': 1e-2}
    elif x.dtype == torch.bfloat16:
        return {'atol': 1e-2, 'rtol': 1e-1}
    else:
        return {'atol': 1e-5, 'rtol': 1e1000}


__test_config__ = ModuleTestConfig(
    competitors=__competitors__,
    reference_competitor='RMSNorm',
    test_cases=[
        {
            'init_args': {'dim': dim},
            'input_args': input_args(dim=dim),
            'output_validator': output_validator,
            'tolerances_fn': tolerances_fn,
            'case_descriptor': f'dim={dim}',
        }
        for dim in dims_to_test
    ]
)


##################################################
################# BENCHMARKING ###################
##################################################


def benchmark_input_provider(init_args: dict) -> tuple:
    """Generates a standard input for benchmarking."""
    dim = init_args.get('dim', 768)
    # Benchmark with larger batch and sequence length
    return (torch.randn(32, 512, dim),)


__benchmark_config__ = BenchmarkConfig(
    module_name='RMSNorm',
    competitors=__competitors__,
    parameter_space={
        'dim': [256, 512, 1024, 2048, 4096], 
    },
    init_arg_builder=lambda params: {'dim': params['dim']},
    input_provider=benchmark_input_provider,
)