from typing import List, Union, Tuple, Any
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.attention.flex_attention import BlockMask, flex_attention, create_block_mask

from modules.base_test_bench_utils import (
    ModuleTestConfig, 
    BenchmarkConfig, 
    Competitor,
    TensorParallelConfig
)
from modules.catalog.fp8_linear import FP8Linear, is_hopper_available
from modules.catalog.norms.rms_norm import RMSNorm
from modules.catalog.sequence_mixing.modded_nanogpt_flex_self_attention import HalfTruncatedRotary


class FlexSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, max_seq_len: int, head_dim=None,
                 fp8: bool = True, device: str = 'cpu', dtype: torch.dtype = torch.float32,):
        super().__init__()
        self.num_heads = num_heads
        if head_dim is None:
            head_dim = dim // num_heads
        self.head_dim = head_dim
        self.scale = 1 / math.sqrt(head_dim)
        hdim = num_heads * head_dim
        std = 0.5 * (dim ** -0.5)
        bound = (3 ** 0.5) * std # improved init scale by @YouJiacheng
        # merged QKV weights: suggested by many, implemented by @fernbear.bsky.social, and further improved by @YouJiacheng
        # https://x.com/hi_tysam/status/1879699187107033311
        self.qkv_w = nn.Parameter(torch.empty(3, hdim, dim).uniform_(-bound, bound))
        self.rotary = HalfTruncatedRotary(head_dim, max_seq_len)
        self.c_proj = FP8Linear(hdim, dim, fp8=fp8, device=device, dtype=dtype)
        self.c_proj.weight.detach().zero_() # zero init suggested by @Grad62304977
        self.norm = RMSNorm()

    def forward(self, x: Tensor, block_mask: BlockMask):
        B, T = x.size(0), x.size(1) # batch size, sequence length
        assert B == 1, "Must use batch size = 1 for FlexAttention"
        q, k, v = F.linear(x, self.qkv_w.flatten(end_dim=1).type_as(x)).view(B, T, 3 * self.num_heads, self.head_dim).chunk(3, dim=-2)
        q, k = self.norm(q), self.norm(k) # QK norm @Grad62304977
        q, k = self.rotary(q), self.rotary(k)
        y = flex_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), 
            block_mask=block_mask, 
            scale=self.scale
        ).transpose(1, 2)
        y = y.contiguous().view(B, T, self.num_heads * self.head_dim) # re-assemble all head outputs side by side
        y = self.c_proj(y)
        return y
    

def run_filter(inputs: Union[torch.Tensor, Tuple[Any]]) -> bool:
    if 'cuda' in str(inputs[0].device):
        return True
    return False


##################################################
#################### TESTING ####################
##################################################


def causal(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx

block_mask = lambda seq: create_block_mask(causal, B=None, H=None, Q_LEN=seq, KV_LEN=seq)


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
    #block_mask = inputs[1]
    output_tensor = outputs[0]
    assert output_tensor.shape == input_tensor.shape, f"Expected output shape {input_tensor.shape}, but got {output_tensor.shape}"
    assert output_tensor.dtype == input_tensor.dtype
    

__competitors__ = {
    'FlexSelfAttention': Competitor(module_class=FlexSelfAttention, run_filter=run_filter),
}


fsa_dims_to_test = [256]
fsa_num_heads_to_test = [4]
fsa_seq_lens_to_test = [2048, 8192]
fsa_dtypes_to_test = [torch.float32, torch.float16, torch.bfloat16]


__test_config__ = ModuleTestConfig(
    competitors=__competitors__,
    reference_competitor='FlexSelfAttention',
    test_cases=[
        {
            'init_args': {
                'dim': dim, 
                'num_heads': num_heads, 
                'max_seq_len': max_seq_len, 
                'fp8': fp8, 
                'dtype': dtype
            },
            'input_args': lambda dev, d=dim, nh=num_heads, s=max_seq_len, fp8=fp8, dt=dtype: (
                torch.randn(1, s, d, device=dev, dtype=dt, requires_grad=True), # x
                block_mask(s),
            ),
            'output_validator': output_validator,
            'tolerances': {'atol': 1e-2, 'rtol': 1e-2}, # Optional
            'case_descriptor': f'dim={dim}_num_heads={num_heads}_max_seq_len={max_seq_len}_fp8={fp8}_dtype={dtype}',
        }
        for dim in fsa_dims_to_test
        for num_heads in fsa_num_heads_to_test
        for max_seq_len in fsa_seq_lens_to_test
        for dtype in fsa_dtypes_to_test
        for fp8 in ([True, False] if is_hopper_available() else [False])
    ]
)


##################################################
################# BENCHMARKING ###################
##################################################


def benchmark_input_provider(init_args: dict, device: str) -> tuple:
    return (
        torch.randn(1, init_args['max_seq_len'], init_args['dim'], device=device, dtype=init_args['dtype']),
        block_mask
    )


__benchmark_config__ = BenchmarkConfig(
    module_name='FlexSelfAttention',
    competitors=__competitors__,
    parameter_space={
        'dim': [256, 512, 1024, 2048],
        'max_seq_len': [1024, 4096, 16_384],
        'num_heads': [4],
        'dtype': [torch.float16, torch.bfloat16, torch.float32],
    },
    init_arg_builder=lambda params: {
        'dim': params['dim'],
        'max_seq_len': params['max_seq_len'],
        'num_heads': params['num_heads'],
        'dtype': params['dtype'],
    },
    input_provider=benchmark_input_provider,
)