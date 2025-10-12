from typing import Tuple, Any

import torch
import torch.nn as nn
from torch.nn import functional as F

from gpt_lab.nn_modules.catalog_utils import ModuleTestConfig, BenchmarkConfig, Competitor


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, dropout: float = 0.0, bias: bool = True):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=bias)
        # output projection
        self.c_proj = nn.Linear(n_embd, n_embd, bias=bias)
        # regularization
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        # efficient attention using Flash Attention CUDA kernels
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)

        """
        # Manual implementation of attention for educational purposes:
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # Create causal mask; normally you'd register the mask as a buffer to save compute
        causal_mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
        att = att.masked_fill(causal_mask == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        """
        
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


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


def causal_attention_run_filter(inputs: Tuple[Any]) -> bool:
    """
    Skip non-float32 on MPS due to known issues with scaled_dot_product_attention.
    MPS backend has compatibility issues with attention operations in float16/bfloat16.
    """
    if len(inputs) > 0 and isinstance(inputs[0], torch.Tensor):
        if 'mps' in str(inputs[0].device) and inputs[0].dtype != torch.float32:
            return False
    return True


__competitors__ = {
    'CausalSelfAttention': Competitor(module_class=CausalSelfAttention, run_filter=causal_attention_run_filter),
}


n_embd_to_test = [256, 768]
n_head_to_test = [4, 8]
seq_len_to_test = [64, 256]
dtypes_to_test = [torch.float32, torch.float16, torch.bfloat16]


def input_args(device: str, n_embd: int, seq_len: int, dtype: torch.dtype):
    batch_size = 4
    return (torch.randn(batch_size, seq_len, n_embd, device=device, dtype=dtype, requires_grad=True),)


def tolerances(dtype: torch.dtype) -> dict:
    if dtype == torch.float32:
        return {'atol': 1e-4, 'rtol': 1e-3}
    elif dtype == torch.float16:
        return {'atol': 5e-3, 'rtol': 1e-1}
    elif dtype == torch.bfloat16:
        return {'atol': 1e-2, 'rtol': 1e-1}
    else:
        return {'atol': 1e-4, 'rtol': 1e-3}


__test_config__ = ModuleTestConfig(
    competitors=__competitors__,
    reference_competitor='CausalSelfAttention',
    test_cases=[
        {
            'init_args': {'n_embd': n_embd, 'n_head': n_head, 'dropout': 0.0, 'bias': True},
            'input_args': lambda dev, n_embd=n_embd, seq_len=seq_len, dt=dt: input_args(device=dev, n_embd=n_embd, seq_len=seq_len, dtype=dt),
            'output_validator': output_validator,
            'tolerances': tolerances(dt),
            'case_descriptor': f'n_embd={n_embd}_n_head={n_head}_seq_len={seq_len}_dtype={dt}',
        }
        for n_embd in n_embd_to_test
        for n_head in n_head_to_test
        for seq_len in seq_len_to_test
        for dt in dtypes_to_test
        if n_embd % n_head == 0  # Only test valid head configurations
    ]
)


##################################################
################# BENCHMARKING ###################
##################################################


def benchmark_input_provider(init_args: dict, device: str) -> tuple:
    """Generates a standard input for benchmarking."""
    n_embd = init_args.get('n_embd', 768)
    seq_len = init_args.get('seq_len', 512)
    dtype = init_args.get('dtype', torch.float32)
    batch_size = 8
    return (torch.randn(batch_size, seq_len, n_embd, device=device, dtype=dtype),)


__benchmark_config__ = BenchmarkConfig(
    module_name='CausalSelfAttention',
    competitors=__competitors__,
    parameter_space={
        'n_embd': [256, 512, 1024, 2048],
        'n_head': [8, 16],
        'seq_len': [256, 512, 1024, 2048],
        'dtype': [torch.float16, torch.bfloat16, torch.float32],
    },
    init_arg_builder=lambda params: {
        'n_embd': params['n_embd'],
        'n_head': params['n_head'],
        'dropout': 0.0,
        'bias': True,
        'seq_len': params['seq_len'],
        'dtype': params['dtype'],
    },
    input_provider=benchmark_input_provider,
)
