from typing import Tuple, Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from nn_modules.catalog_utils import ModuleTestConfig, BenchmarkConfig, Competitor


torch.set_float32_matmul_precision('medium')
torch._dynamo.config.recompile_limit = 100


"""
FP8 matmul by @YouJiacheng
NOTE: Only works on Hopper GPUs; others will still pass tests but only because they skipped fp8=True case
"""


def is_hopper_available() -> bool:
    """Checks if a Hopper architecture GPU is available."""
    if not torch.cuda.is_available():
        return False
    # Hopper GPUs have compute capability 9.0 or higher
    return any(torch.cuda.get_device_capability(i)[0] >= 9 for i in range(torch.cuda.device_count()))


@torch.library.custom_op("nanogpt::mm", mutates_args=())
def mm_op(x: Tensor, w: Tensor, x_s: float, w_s: float, grad_s: float) -> tuple[Tensor, Tensor, Tensor]:
    @torch.compile
    def impl(x: Tensor, w: Tensor):
        assert x.is_contiguous() and w.is_contiguous()
        x_f8 = x.div(x_s).to(torch.float8_e4m3fn)
        w_f8 = w.div(w_s).to(torch.float8_e4m3fn)
        out = torch._scaled_mm(
            x_f8,
            w_f8.T,
            out_dtype=torch.bfloat16,
            scale_a=x.new_tensor(x_s, dtype=torch.float32),
            scale_b=x.new_tensor(w_s, dtype=torch.float32),
            use_fast_accum=True,
        )
        return out, x_f8, w_f8

    return impl(x, w)

@mm_op.register_fake
def _(x: Tensor, w: Tensor, *_):
    assert x.ndim == w.ndim == 2
    assert x.shape[1] == w.shape[1]
    assert x.device == w.device
    assert x.is_contiguous() and w.is_contiguous()
    return x @ w.T, x.to(torch.float8_e4m3fn), w.to(torch.float8_e4m3fn)

@torch.library.custom_op("nanogpt::mm_backward", mutates_args=())
def mm_backward_op(g: Tensor, x_f8: Tensor, w_f8: Tensor, x_s: float, w_s: float, grad_s: float) -> tuple[Tensor, Tensor]:
    @torch.compile
    def impl(grad: Tensor, x_f8: Tensor, w_f8: Tensor):
        assert grad.is_contiguous()
        x_inv_s = grad.new_tensor(x_s, dtype=torch.float32)
        w_inv_s = grad.new_tensor(w_s, dtype=torch.float32)
        grad_inv_s = grad.new_tensor(grad_s, dtype=torch.float32)
        grad_f8 = grad.div(grad_s).to(torch.float8_e5m2)
        grad_x = torch._scaled_mm(
            grad_f8,
            w_f8.T.contiguous().T,
            out_dtype=torch.bfloat16,
            scale_a=grad_inv_s,
            scale_b=w_inv_s,
            use_fast_accum=False,
        )
        # faster than grad_f8_t @ x_f8, for (d_out, d_in) == (50304, 768)
        grad_w = torch._scaled_mm(
            x_f8.T.contiguous(),
            grad_f8.T.contiguous().T,
            out_dtype=torch.float32,
            scale_a=x_inv_s,
            scale_b=grad_inv_s,
            use_fast_accum=False,
        ).T
        return grad_x, grad_w

    return impl(g, x_f8, w_f8)

@mm_backward_op.register_fake
def _(g: Tensor, x_f8: Tensor, w_f8: Tensor, *_):
    return x_f8.to(torch.bfloat16), w_f8.T.contiguous().T.to(torch.float32)

def backward(ctx, grad_out: Tensor, *_):
    x_f8, w_f8 = ctx.saved_tensors
    x_s, w_s, grad_s = ctx.scales
    grad_x, grad_w = torch.ops.nanogpt.mm_backward(
        grad_out, x_f8, w_f8, x_s, w_s, grad_s
    )
    return grad_x, grad_w, None, None, None

def setup_context(ctx: torch.autograd.function.FunctionCtx, inputs, output):
    *_, x_s, w_s, grad_s = inputs
    _, x_f8, w_f8 = output
    ctx.save_for_backward(x_f8, w_f8)
    ctx.scales = x_s, w_s, grad_s
    ctx.set_materialize_grads(False)

mm_op.register_autograd(backward, setup_context=setup_context)


class FP8Linear(nn.Linear):
    def __init__(self, in_features: int, out_features: int,  
                 fp8: bool, x_s: float = 1.0, w_s: float = 1.0, grad_s: float = 1.0,
                 device: str = None, dtype: torch.dtype = torch.float32,):
        super().__init__(in_features, out_features, bias=False, device=device, dtype=dtype)
        self.fp8 = fp8 and is_hopper_available()
        self.x_s = x_s
        self.w_s = w_s
        self.grad_s = grad_s

    def reset_parameters(self) -> None:
        std = 0.5 * (self.in_features ** -0.5) # 0.5 is a bit better than the default 1/sqrt(3)
        bound = (3 ** 0.5) * std
        with torch.no_grad():
            self.weight.uniform_(-bound, bound)

    def forward(self, x: Tensor):
        if self.fp8 and self.training:
            _x = x.flatten(0, -2)
            out: Tensor = torch.ops.nanogpt.mm(_x, self.weight, x_s=self.x_s, w_s=self.w_s, grad_s=self.grad_s)[0]
            return out.reshape(*x.shape[:-1], -1)
        else:
            return F.linear(x, self.weight.type_as(x))


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
    expected_shape = (*input_tensor.shape[:-1], module.out_features)
    assert output_tensor.shape == expected_shape, f"Expected output shape {expected_shape}, but got {output_tensor.shape}"
    assert output_tensor.dtype == input_tensor.dtype
    

__competitors__ = {
    'FP8Linear': Competitor(module_class=FP8Linear),
}


def input_args(dev, in_features, out_features, dtype):
    return (torch.randn(2048, in_features, device=dev, dtype=torch.float32, requires_grad=True),)

__test_config__ = ModuleTestConfig(
    competitors=__competitors__,
    reference_competitor='FP8Linear',
    test_cases=[
        {
            'init_args': {
                'in_features': in_features, 
                'out_features': out_features, 
                'fp8': fp8,
                'dtype': dtype,
                },
            'input_args': lambda dev, in_dim=in_features, out_dim=out_features, dt=dtype: input_args(dev, in_dim, out_dim, dt),
            'output_validator': output_validator,
            'tolerances': {'atol': 1e-1, 'rtol': 1},          # Optional
            'case_descriptor': f'(in_features,out_features)=({in_features},{out_features})_fp8={fp8}_dtype={dtype}',
        }
        for in_features, out_features in [(128, 128), (512, 2048), (1832, 4271)]
        for fp8 in ([True, False] if is_hopper_available() else [False])
        for dtype in [torch.float16, torch.bfloat16, torch.float32]
    ]
)


##################################################
################# BENCHMARKING ###################
##################################################


def benchmark_input_provider(init_args: dict, device: str) -> tuple:
    """Generates a standard input for benchmarking."""
    # input shape: (batch_size, sequence_length, dimension)
    dtype = init_args.get('dtype', torch.float32)
    return (torch.randn(1, 1, init_args['in_features'], device=device, dtype=torch.float32),)

__benchmark_config__ = BenchmarkConfig(
    module_name='FP8Linear',
    competitors=__competitors__,
    parameter_space={
        'dim': [32, 64, 128, 512, 1024, 2048, 4096],
        'dtype': [torch.float16, torch.bfloat16, torch.float32],
        'fp8': [True, False] if is_hopper_available() else [False],
    },
    init_arg_builder=lambda params: {
        'in_features': params['dim'],
        'out_features': params['dim'],
        'fp8': params['fp8'],
    },
    input_provider=benchmark_input_provider,
)