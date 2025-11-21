from typing import List, Union, Tuple, Any
import torch
import torch.nn as nn

from gpt_lab.nn_modules.catalog_utils import (
    ModuleTestConfig, 
    BenchmarkConfig, 
    Competitor,
)


##################################################
############# PRIMARY PYTORCH MODULE #############
##################################################


class EfficientDropPath(nn.Module):
    """
    Efficient Drop Path (Stochastic Depth) that skips computation for dropped samples.
    Wraps a module and only executes it on kept batch elements.
    
    Constraints:
    - Only works along the batch dimension (dim 0).
    - Drops a constant number of samples per batch (floor(B * drop_prob)).
    - Input shape must be (B, ...). Packed sequences (1, Tokens, D) will likely result in no drops
      unless drop_prob >= 1.0, effectively disabling drop path for that mode (which is intended behavior
      if we only drop at batch level).
    """
    def __init__(self, module: nn.Module, drop_prob: float = 0.):
        super().__init__()
        self.module = module
        self.drop_prob = drop_prob

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.:
            return self.module(x)

        B = x.shape[0]
        # Determine number of samples to keep
        # We ensure deterministic count for a given B
        n_drop = int(B * self.drop_prob)
        
        # If we can't drop at least one sample, we process everything.
        if n_drop == 0:
            return self.module(x)
            
        n_keep = B - n_drop
        
        # Select indices to keep
        perm = torch.randperm(B, device=x.device)
        keep_indices = perm[:n_keep]
        
        # Gather inputs
        x_kept = x[keep_indices]
        
        # Compute only on kept samples
        out_kept = self.module(x_kept)
        
        # Prepare output
        # We assume output shape has same non-batch dims as out_kept
        out_shape = (B,) + out_kept.shape[1:]
        output = out_kept.new_zeros(out_shape)
        
        # Scatter results back
        output[keep_indices] = out_kept
        
        # Scale to maintain expected value
        # The effective keep rate is n_keep / B
        scale = B / n_keep
        output.mul_(scale)
        
        return output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(x)

    def extra_repr(self) -> str:
        return f'drop_prob={self.drop_prob}'


########################################################
# PRECOMPILED IMPLEMENTATION FOR TESTING torch.compile #
########################################################


class PreCompiledEfficientDropPath(EfficientDropPath):
    def __init__(self, module: nn.Module, drop_prob: float = 0.):
        super().__init__(module, drop_prob)
        # Compile the implementation
        self._forward_impl = torch.compile(self._forward_impl, mode='default')


def pre_compiled_run_filter(inputs: Union[torch.Tensor, Tuple[Any]]) -> bool:
    """
    Ensure that testing is only attempted on GPU for torch.compile.
    """
    if len(inputs) > 0 and isinstance(inputs[0], torch.Tensor):
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
    Validates whether the output shape and type matches the input.
    Also verifies the statistical properties and scaling logic.
    """
    input_tensor = inputs[0] 
    output_tensor = outputs[0]
    
    # Check shape (assuming module is Identity for validation tests)
    assert output_tensor.shape == input_tensor.shape, f"Expected output shape {input_tensor.shape}, but got {output_tensor.shape}"
    assert output_tensor.dtype == input_tensor.dtype
    assert output_tensor.device == input_tensor.device
    
    if not module.training or module.drop_prob == 0.:
        # Should be identity if module is identity
        torch.testing.assert_close(output_tensor, input_tensor)
        return

    B = input_tensor.shape[0]
    n_drop = int(B * module.drop_prob)
    
    if n_drop == 0:
        torch.testing.assert_close(output_tensor, input_tensor)
        return

    n_keep = B - n_drop
    
    # Verify that exactly n_drop samples are zero
    flattened = output_tensor.reshape(B, -1)
    sample_is_zero = (flattened.abs().sum(dim=1) == 0)
    num_zeros = sample_is_zero.sum().item()
    
    assert num_zeros == n_drop, f"Expected {n_drop} dropped samples, got {num_zeros}"
    
    # Verify scaling for kept paths
    scale_factor = B / n_keep
    
    kept_mask = ~sample_is_zero
    if kept_mask.any():
        scaled_input = input_tensor * scale_factor
        
        # Use relaxed tolerance for float arithmetic differences
        tol = 1e-4
        if output_tensor.dtype in (torch.float16, torch.bfloat16):
            tol = 1e-2
            
        torch.testing.assert_close(
            output_tensor[kept_mask], 
            scaled_input[kept_mask],
            rtol=tol, atol=tol
        )


# Use Identity as the inner module for correctness testing so we can predict the output
__competitors__ = {
    'EfficientDropPath': Competitor(
        module_class=lambda **kwargs: EfficientDropPath(module=nn.Identity(), **kwargs)
    ),
    'PreCompiledEfficientDropPath': Competitor(
        module_class=lambda **kwargs: PreCompiledEfficientDropPath(module=nn.Identity(), **kwargs),
        run_filter=pre_compiled_run_filter
    ),
}


def input_args(shape: Tuple[int]):
    return (torch.randn(*shape, requires_grad=True),)

shapes_to_test = [
    (4, 128, 768),  # (B, N, D)
    (1, 1024, 768), # (1, N, D) - Should drop nothing with small prob
    (10, 768),      # (B, D)
]
probs_to_test = [0.0, 0.1, 0.5]


__test_config__ = ModuleTestConfig(
    competitors=__competitors__,
    reference_competitor='EfficientDropPath',
    test_cases=[
        {
            'init_args': {'drop_prob': prob},
            'input_args': input_args(shape),
            'output_validator': output_validator,
            # High tolerance for randomness check not needed here since we validate logic explicitly
            # But we provide a dummy one just in case
            'tolerances_fn': lambda x: {'atol': 1e6, 'rtol': 1e6}, 
            'case_descriptor': f'shape={shape}_prob={prob}',
        }
        for shape in shapes_to_test
        for prob in probs_to_test
    ]
)


##################################################
################# BENCHMARKING ###################
##################################################


def benchmark_input_provider(init_args: dict) -> tuple:
    """Generates a standard input for benchmarking."""
    shape = init_args.get('shape', (128, 128, 768))
    return (torch.randn(*shape),)


class BenchmarkHelper(nn.Module):
    """Helper to make EfficientDropPath work with BenchmarkConfig which expects flat kwargs"""
    def __init__(self, drop_prob: float, dim: int, drop_path_cls=EfficientDropPath):
        super().__init__()
        # Use a somewhat heavy module to show benefit of skipping compute
        self.model = drop_path_cls(
            module=nn.Sequential(
                nn.Linear(dim, dim),
                nn.SiLU(),
                nn.Linear(dim, dim)
            ),
            drop_prob=drop_prob
        )
    
    def forward(self, x):
        return self.model(x)

__benchmark_config__ = BenchmarkConfig(
    module_name='EfficientDropPath',
    competitors={
        'EfficientDropPath': Competitor(module_class=BenchmarkHelper),
        'PreCompiledEfficientDropPath': Competitor(
            module_class=lambda **kwargs: BenchmarkHelper(drop_path_cls=PreCompiledEfficientDropPath, **kwargs),
            run_filter=pre_compiled_run_filter
        ),
    },
    parameter_space={
        'shape': [(128, 128, 768), (32, 1024, 768)],
        'drop_prob': [0.0, 0.1, 0.5, 0.9],
    },
    init_arg_builder=lambda params: {
        'drop_prob': params['drop_prob'],
        'dim': params['shape'][-1],
    },
    input_provider=benchmark_input_provider,
)

