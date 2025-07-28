import math

import torch
import torch.nn as nn
import triton
import triton.language as tl


autotune_configs = [
    triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE': 8}, num_stages=3, num_warps=8),
    triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE': 8}, num_stages=4, num_warps=4),
    #triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE': 8}, num_stages=4, num_warps=4),
    #triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE': 8}, num_stages=4, num_warps=4),
    #triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE': 8}, num_stages=4, num_warps=4),
    #triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE': 8}, num_stages=4, num_warps=4),
    #triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE': 8}, num_stages=5, num_warps=2),
    #triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE': 8}, num_stages=5, num_warps=2),
]
@triton.autotune(configs = autotune_configs, key=['M', 'N', 'K'])
@triton.jit
def _matmul(
    a_ptr, b_ptr, c_ptr, 
    M, N, K, 
    stride_a_M, stride_a_K, 
    stride_b_K, stride_b_N, 
    stride_c_M, stride_c_N, 
    # meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE: tl.constexpr, 
):
    PID = tl.program_id(axis=0) 
    num_PID_along_M = tl.cdiv(M, BLOCK_SIZE_M)
    num_PID_along_N = tl.cdiv(N, BLOCK_SIZE_N)
    num_PID_in_group = GROUP_SIZE * num_PID_along_N
    group_id = PID // num_PID_in_group 
    first_PID_in_group_along_M = group_id * GROUP_SIZE 
    group_size_adj = min(num_PID_along_M - first_PID_in_group_along_M, GROUP_SIZE) 
    PID_M = first_PID_in_group_along_M + ((PID % num_PID_in_group) % group_size_adj)
    PID_N = (PID % num_PID_in_group) // group_size_adj
    
    offsets_M = PID_M * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offsets_N = PID_N * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offsets_K = tl.arange(0, BLOCK_SIZE_K)
    a_offsets = offsets_M[:, None] * stride_a_M + offsets_K[None, :] * stride_a_K
    b_offsets = offsets_K[:, None] * stride_b_K + offsets_N[None, :] * stride_b_N

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        mask = offsets_K < K - k * BLOCK_SIZE_K
        a = tl.load(a_ptr + a_offsets, mask=mask[None, :], other=0.0)
        b = tl.load(b_ptr + b_offsets, mask=mask[:, None], other=0.0) 
        accumulator = tl.dot(a, b, acc=accumulator)
        a_offsets += BLOCK_SIZE_K * stride_a_K
        b_offsets += BLOCK_SIZE_K * stride_b_K

    accumulator = accumulator.to(tl.float16)

    c_offsets = stride_c_M * offsets_M[:, None] + stride_c_N * offsets_N[None, :]
    c_mask = (offsets_M[:, None] < M) & (offsets_N[None, :] < N) 
    tl.store(c_ptr + c_offsets, accumulator, mask=c_mask) 


autotune_configs = [
    triton.Config({'BLOCK_SIZE_M': BSM, 'BLOCK_SIZE_K': BSK, 'BLOCK_SIZE_N': BSN, 'GROUP_SIZE': GS}, num_stages=ns, num_warps=nw)
    for BSM in [32]#, 64, 128]
    for BSK in [32]#, 64, 128, 256]
    for BSN in [32]#, 64]
    for GS in [8]
    for ns in [3]#, 4, 5]
    for nw in [2]#, 4, 8]
]
@triton.autotune(configs = autotune_configs, key=['M', 'N', 'K'])#, 'dtype'])
@triton.jit
def _fused_GLU(
    x_ptr, wup_ptr, wgate_ptr, 
    up_ptr, gate_ptr, gate_act_ptr, hidden_ptr,
    stride_x_M, stride_x_K,
    stride_wup_K, stride_wup_N, 
    stride_wgate_K, stride_wgate_N, 
    stride_up_M, stride_up_N,
    stride_gate_M, stride_gate_N,
    stride_gate_act_M, stride_gate_act_N,
    stride_hidden_M, stride_hidden_N,
    M, K, N,
    #dtype,
    act: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    PID = tl.program_id(axis=0) 
    num_PID_along_M = tl.cdiv(M, BLOCK_SIZE_M)
    num_PID_along_N = tl.cdiv(N, BLOCK_SIZE_N)
    num_PID_in_group = GROUP_SIZE * num_PID_along_N
    group_id = PID // num_PID_in_group 
    first_PID_in_group_along_M = group_id * GROUP_SIZE 
    group_size_adj = min(num_PID_along_M - first_PID_in_group_along_M, GROUP_SIZE) 
    PID_M = first_PID_in_group_along_M + ((PID % num_PID_in_group) % group_size_adj)
    PID_N = (PID % num_PID_in_group) // group_size_adj
    
    offsets_M = PID_M * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offsets_N = PID_N * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offsets_K = tl.arange(0, BLOCK_SIZE_K)

    x_offsets = offsets_M[:, None] * stride_x_M + offsets_K[None, :] * stride_x_K
    wup_offsets = offsets_K[:, None] * stride_wup_K + offsets_N[None, :] * stride_wup_N
    wgate_offsets = offsets_K[:, None] * stride_wgate_K + offsets_N[None, :] * stride_wgate_N

    up_accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    gate_accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        mask = offsets_K < K - k * BLOCK_SIZE_K
        x = tl.load(x_ptr + x_offsets, mask=mask[None, :], other=0.0)
        wup = tl.load(wup_ptr + wup_offsets, mask=mask[:, None], other=0.0) 
        wgate = tl.load(wgate_ptr + wgate_offsets, mask=mask[:, None], other=0.0) 
        up_accumulator = tl.dot(x, wup, acc=up_accumulator)
        gate_accumulator = tl.dot(x, wgate, acc=gate_accumulator)
        x_offsets += BLOCK_SIZE_K * stride_x_K
        wup_offsets += BLOCK_SIZE_K * stride_wup_K
        wgate_offsets += BLOCK_SIZE_K * stride_wgate_K
    
    up_offsets = stride_up_M * offsets_M[:, None] + stride_up_N * offsets_N[None, :]
    up_mask = (offsets_M[:, None] < M) & (offsets_N[None, :] < N)
    tl.store(up_ptr + up_offsets, up_accumulator.to(up_ptr.type.element_ty), mask=up_mask)
    gate_offsets = stride_gate_M * offsets_M[:, None] + stride_gate_N * offsets_N[None, :]
    gate_mask = (offsets_M[:, None] < M) & (offsets_N[None, :] < N)
    tl.store(gate_ptr + gate_offsets, gate_accumulator.to(gate_ptr.type.element_ty), mask=gate_mask)

    relu_lower: tl.constexpr = 0
    relu_upper: tl.constexpr = 1e6
    if act == "relu2":
        gate_accumulator = tl.clamp(gate_accumulator, relu_lower, relu_upper)
        gate_accumulator *= gate_accumulator
    elif act == "silu":
        gate_accumulator = gate_accumulator * (1 / (1 + tl.exp(-gate_accumulator)))
    else: # defaults to relu
        gate_accumulator = tl.clamp(gate_accumulator, relu_lower, relu_upper)
    gate_act = gate_accumulator

    gate_act_offsets = stride_gate_act_M * offsets_M[:, None] + stride_gate_act_N * offsets_N[None, :]
    gate_act_mask = (offsets_M[:, None] < M) & (offsets_N[None, :] < N)
    tl.store(gate_act_ptr + gate_act_offsets, gate_act.to(gate_act_ptr.type.element_ty), mask=gate_act_mask)

    hidden = up_accumulator * gate_act

    hidden_offsets = stride_hidden_M * offsets_M[:, None] + stride_hidden_N * offsets_N[None, :]
    hidden_mask = (offsets_M[:, None] < M) & (offsets_N[None, :] < N)
    tl.store(hidden_ptr + hidden_offsets, hidden.to(hidden_ptr.type.element_ty), mask=hidden_mask)


@torch.compile
def gated_mlp_bwd(x, Wup, Wgate, Wdown, up, gate, gate_act, hidden, out, dLdout, act: str):
    dLdhidden = dLdout @ Wdown.T # (B, D) @ (D, H) -> (B, H)
    dLdWdown = hidden.T @ dLdout # (H, B) @ (B, D) -> (H, D)

    dLdgate_act = dLdhidden * up # (B, D) * (B, D) -> (B, D)
    dLdup = dLdhidden * gate # (B, D) * (B, D) -> (B, D)

    if act == "relu2":
        dLdgate = torch.where(gate > 0, 2 * up * dLdgate_act, torch.zeros_like(gate)) # (B, D)
    elif act == "silu":
        sigmoid = 1 / (1 + torch.exp(-up))
        dLdgate = (sigmoid + gate * sigmoid * (1 - sigmoid)) * dLdgate_act # (B, D)
    elif act == "relu": # relu
        dLdgate = torch.where(gate > 0, dLdgate_act, torch.zeros_like(gate)) # (B, D)
    else:
        raise ValueError()
        
    dLdx_up_part = dLdup @ Wup.T # (B, H) @ (H, D) -> (B, D)
    dLdWup = x.T @ dLdup # (D, B) @ (B, H) -> (D, H)

    dLdx_gate_part = dLdgate @ Wgate.T # (B, H) @ (H, D) -> (B, D)
    dLdWgate = x.T @ dLdgate # (D, B) @ (B, H) -> (D, H)

    dLdx = dLdx_up_part + dLdx_gate_part # (B, D)

    return dLdx, dLdWup, dLdWgate, dLdWdown


class _GatedMLPKernelFn(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, x: torch.Tensor, Wup: torch.Tensor, Wgate: torch.Tensor, Wdown: torch.Tensor, act: str):
        assert x.device == Wup.device == Wgate.device == Wdown.device
        #assert x.is_contiguous() and Wup.is_contiguous() and Wgate.is_contiguous() and Wdown.is_contiguous()
        assert Wup.ndim == Wgate.ndim == Wdown.ndim == 2
        assert Wup.shape == Wgate.shape
        assert Wup.shape[1] == Wdown.shape[0]
        assert x.shape[-1] == Wup.shape[0] == Wdown.shape[1]

        if x.ndim == 1:
            x = x.view(1, x.shape[0])
        if x.ndim == 3:
            x = x.view(x.shape[:-1], x.shape[-1])
        B = x.shape[0]
        D = x.shape[1]
        H = Wup.shape[1]

        up = torch.empty((B, H), device=x.device, dtype=x.dtype)
        gate = torch.empty((B, H), device=x.device, dtype=x.dtype)
        gate_act = torch.empty((B, H), device=x.device, dtype=x.dtype)
        hidden = torch.empty((B, H), device=x.device, dtype=x.dtype)

        grid = lambda meta: (triton.cdiv(B, meta['BLOCK_SIZE_M']) * triton.cdiv(H, meta['BLOCK_SIZE_N']), B)
        _fused_GLU[grid](
            x, Wup, Wgate, 
            up, gate, gate_act, hidden,
            x.stride(0), x.stride(1),
            Wup.stride(0), Wup.stride(1),
            Wgate.stride(0), Wgate.stride(1),
            up.stride(0), up.stride(1),
            gate.stride(0), gate.stride(1),
            gate_act.stride(0), gate_act.stride(1),
            hidden.stride(0), hidden.stride(1),
            M=B, K=D, N=H,
            #dtype=x.dtype,
            act=act,
        )

        out = hidden @ Wdown
        
        ctx.save_for_backward(x, Wup, Wgate, Wdown, up, gate, gate_act, hidden, out)
        ctx.act = act

        return out
    
    @staticmethod
    def backward(ctx, dLdout):
        x, Wup, Wgate, Wdown, up, gate, gate_act, hidden, out = ctx.saved_tensors
        act = ctx.act
        dLdx, dLdWup, dLdWgate, dLdWdown = gated_mlp_bwd(x, Wup, Wgate, Wdown, up, gate, gate_act, hidden, out, dLdout, act)
        return dLdx, dLdWup, dLdWgate, dLdWdown, None
    

"""
matmul derivatives:
A        B       C
(m,k) @ (k,n) = (m,n)

dLdC     B^T    dLdA
(m,n) @ (n,k) = (m,k)

A^T      dLdC    dLdB
(k,m) @ (m,n) = (k,n)
"""