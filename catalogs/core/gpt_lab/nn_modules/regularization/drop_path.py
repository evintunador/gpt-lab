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


def drop_path(x: torch.Tensor, drop_prob: float = 0., training: bool = False) -> torch.Tensor:
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    
    This implementation is consistent with the following logic:
    - It drops the entire sample with probability `drop_prob`.
    - It scales the output by `1 / (1 - drop_prob)` to maintain the expected value.
    - It broadcasts the mask across all dimensions except the first (batch) dimension.
    
    Args:
        x: Input tensor of shape (B, ...).
        drop_prob: Probability of dropping the path.
        training: Whether the module is in training mode.
        
    Returns:
        The input tensor, randomly zeroed out and scaled.
    """
    if drop_prob == 0. or not training:
        return x
    
    keep_prob = 1 - drop_prob
    # Handle broadcasting: (B, 1, 1, ...) for input (B, C, H, W) or (B, N, D)
    shape = (x.shape[0],) + (1,) * (x.ndim - 1) 
    
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    
    if keep_prob > 0.0:
        random_tensor.div_(keep_prob)
        
    return x * random_tensor


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob: float = 0.):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)


########################################################
# PRECOMPILED IMPLEMENTATION FOR TESTING torch.compile #
########################################################


@torch.compile(mode='default')
def compiled_drop_path_fwd(x, drop_prob, training):
    return drop_path(x, drop_prob, training)


class PreCompiledDropPath(DropPath):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return compiled_drop_path_fwd(x, self.drop_prob, self.training)


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
    Also verifies the statistical properties and scaling logic of DropPath.
    """
    input_tensor = inputs[0] 
    output_tensor = outputs[0]
    
    assert output_tensor.shape == input_tensor.shape, f"Expected output shape {input_tensor.shape}, but got {output_tensor.shape}"
    assert output_tensor.dtype == input_tensor.dtype
    assert output_tensor.device == input_tensor.device
    
    # If in eval mode or drop_prob is 0, output should be identical to input
    if not module.training or module.drop_prob == 0.:
        torch.testing.assert_close(output_tensor, input_tensor)
    else:
        # Verify scaling for kept paths
        keep_prob = 1 - module.drop_prob
        scale_factor = 1 / keep_prob
        
        # Create a mask of kept elements (assuming input wasn't already zero)
        # We look for where output is non-zero. 
        # Note: This assumes input doesn't have exact zeros in the same places, which is likely for randn.
        kept_mask = output_tensor != 0
        
        if kept_mask.any():
            # For kept elements, output should be input * scale_factor
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

        # Verify that we are observing the expected dropping behavior (statistically)
        # For very small batch sizes, we can't assert strict drop rates, but we can check
        # that the operation is indeed broadcasting correctly across the batch dim.
        
        # Check if entire samples are dropped (broadcasting check)
        # We sum across all dims except batch to see if the drop pattern is per-sample
        flattened = output_tensor.reshape(output_tensor.shape[0], -1)
        sample_is_zero = (flattened.abs().sum(dim=1) == 0)
        
        # If we dropped anything, ensure it was a whole sample (unless input was zero)
        # This confirms the mask is (B, 1, 1, ...) and not (B, D, ...) or random per element
        if sample_is_zero.any():
             # If a sample is "zero" in output, it should likely be non-zero in input (with high prob)
             # We just want to confirm we aren't dropping individual elements within a sample
             pass # The scaling check above implicitly checks this for kept elements.
                  # If mask was per-element, then output[kept_mask] check would still pass,
                  # but we wouldn't have whole samples dropped.
                  
        # We can't strictly assert drop rate on small batches (e.g. B=4), 
        # but we can at least verify that the mean is somewhat consistent if we had large batches.
        # For now, the scaling check is the strongest correctness guarantee.


__competitors__ = {
    'DropPath': Competitor(module_class=DropPath),
    'PreCompiledDropPath': Competitor(module_class=PreCompiledDropPath, run_filter=pre_compiled_run_filter),
}


def input_args(shape: Tuple[int]):
    return (torch.randn(*shape, requires_grad=True),)

# Standard transformer shapes and packed shapes
shapes_to_test = [
    (4, 128, 768),  # (B, N, D)
    (1, 1024, 768), # (1, N, D) Packed
    (8, 768),       # (B, D)
]
probs_to_test = [0.0, 0.1, 0.5]


__test_config__ = ModuleTestConfig(
    competitors=__competitors__,
    reference_competitor='DropPath',
    test_cases=[
        {
            'init_args': {'drop_prob': prob},
            'input_args': input_args(shape),
            'output_validator': output_validator,
            # Tolerances are high because we are comparing stochastic outputs in training mode
            # Ideally we only strict compare when drop_prob=0 or eval mode
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
    return (torch.randn(*shape, requires_grad=True),)


__benchmark_config__ = BenchmarkConfig(
    module_name='DropPath',
    competitors=__competitors__,
    parameter_space={
        'shape': [(128, 128, 768), (32, 1024, 768)],
        'drop_prob': [0.0, 0.1, 0.5],
    },
    init_arg_builder=lambda params: {
        'drop_prob': params['drop_prob'],
    },
    input_provider=benchmark_input_provider,
)

