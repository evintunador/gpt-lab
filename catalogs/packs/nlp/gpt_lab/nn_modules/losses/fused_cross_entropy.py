from typing import Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from gpt_lab.nn_modules.catalog_utils import ModuleTestConfig, BenchmarkConfig, Competitor

try:
    import triton
    from liger_kernel.transformers.fused_linear_cross_entropy import (
        LigerFusedLinearCrossEntropyLoss,
    )
    LIGER_AVAILABLE = True
except ImportError:
    LIGER_AVAILABLE = False
    # Dummy class to avoid NameError in type hints or if instantiated (will raise in init)
    class LigerFusedLinearCrossEntropyLoss:
        def __init__(self, *args, **kwargs): pass
        def __call__(self, *args, **kwargs): pass


class TorchLinearCELoss(torch.nn.Module):
    """Ground truth implementation of the linear fused with torch based cross entropy loss.

    :param D: hidden size
    :param V: vocab size
    :param ignore_index: index to ignore
    :param weight: optional weight tensor (V, D) for weight tying
    """

    def __init__(self, D: int, V: int, dtype: torch.dtype, ignore_index: int = -100, weight: torch.Tensor = None):
        super().__init__()
        self.weight = weight
        if self.weight is None:
            self.lin = torch.nn.Linear(
                in_features=D, out_features=V, bias=False, dtype=dtype
            )
        else:
            if self.weight.shape != (V, D):
                raise ValueError(f"Expected weight shape ({V}, {D}), got {self.weight.shape}")
            self.lin = None
            
        self.ce_loss = torch.nn.CrossEntropyLoss(
            ignore_index=ignore_index, reduction="mean"
        )

    def forward(self, x, y):
        if self.lin is not None:
            logits = self.lin(x)
        else:
            logits = F.linear(x, self.weight)
        return self.ce_loss(logits, y)


class FusedLinearCELoss(torch.nn.Module):
    def __init__(self, D: int, V: int, dtype: torch.dtype, ignore_index: int = -100, weight: torch.Tensor = None):
        super().__init__()
        if not LIGER_AVAILABLE:
            raise ImportError("liger_kernel is not installed")

        self.weight = weight
        if self.weight is None:
            self.lin = torch.nn.Linear(
                in_features=D, out_features=V, bias=False, dtype=dtype
            )
        else:
            if self.weight.shape != (V, D):
                raise ValueError(f"Expected weight shape ({V}, {D}), got {self.weight.shape}")
            self.lin = None

        self.ce_loss = LigerFusedLinearCrossEntropyLoss(
            ignore_index=ignore_index, reduction="mean"
        )

    def forward(self, x, y):
        w = self.lin.weight if self.lin is not None else self.weight
        return self.ce_loss(w, x, y)


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
    """
    # output should be a scalar loss
    output_tensor = outputs[0]
    assert output_tensor.ndim == 0, f"Expected scalar loss, but got shape {output_tensor.shape}"


def fused_run_filter(inputs: Tuple[Any]) -> bool:
    """
    Liger kernel usually requires CUDA.
    """
    if not LIGER_AVAILABLE:
        return False
        
    if len(inputs) > 0 and isinstance(inputs[0], torch.Tensor):
        if 'cuda' not in str(inputs[0].device):
            return False
    return True


__competitors__ = {
    'TorchLinearCELoss': Competitor(module_class=TorchLinearCELoss),
    'FusedLinearCELoss': Competitor(module_class=FusedLinearCELoss, run_filter=fused_run_filter),
}


def input_args_generator(D, V, dtype):
    batch_size = 4
    seq_len = 32
    # Note: The test harness will move these to the correct device (CUDA/CPU)
    x = torch.randn(batch_size * seq_len, D, dtype=dtype, requires_grad=True)
    y = torch.randint(0, V, (batch_size * seq_len,))
    return (x, y)


__test_config__ = ModuleTestConfig(
    competitors=__competitors__,
    reference_competitor='TorchLinearCELoss',
    test_cases=[
        {
            'init_args': {'D': D, 'V': V, 'dtype': dtype},
            'input_args': input_args_generator(D, V, dtype),
            'output_validator': output_validator,
            'tolerances_fn': lambda x: {'atol': 1e-2, 'rtol': 1e-2},
            'case_descriptor': f'D={D}_V={V}_dtype={dtype}',
        }
        for D in [64]
        for V in [128]
        for dtype in [torch.float32, torch.bfloat16]
    ]
)


##################################################
################# BENCHMARKING ###################
##################################################


def benchmark_input_provider(init_args: dict) -> tuple:
    D = init_args['D']
    V = init_args['V']
    dtype = init_args['dtype']
    batch_size = 8
    seq_len = 1024
    
    x = torch.randn(batch_size * seq_len, D, dtype=dtype, requires_grad=True)
    y = torch.randint(0, V, (batch_size * seq_len,))
    return (x, y)


__benchmark_config__ = BenchmarkConfig(
    module_name='LinearCELoss',
    competitors=__competitors__,
    parameter_space={
        'D': [768, 2048, 4096],
        'V': [32000, 50257],
        'dtype': [torch.bfloat16],
    },
    init_arg_builder=lambda params: {
        'D': params['D'],
        'V': params['V'],
        'dtype': params['dtype'],
    },
    input_provider=benchmark_input_provider,
)
