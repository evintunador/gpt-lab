import torch
from torch.optim import Optimizer
import torch.distributed as dist
import torch.nn as nn

from optimizers.bulk_test_bench_utils import OptimizerConfig


class Muon(Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    https://kellerjordan.github.io/posts/muon/

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings:
    - This optimizer should not be used for the embedding layer, the final fully connected layer,
    or any {0,1}-D parameters; those should all be optimized by a standard method (e.g., AdamW).
    - To use it with 4D convolutional filters, it works well to just flatten their last 3 dimensions.

    Arguments:
        lr: The learning rate used by the internal SGD.
        momentum: The momentum used by the internal SGD.
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        ns_steps: The number of Newton-Schulz iteration steps to use.
    """
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5, rank=0, world_size=1):
        self.rank = rank
        self.world_size = world_size
        
        params: list[torch.tensor] = [*params]
        for param in params:
            assert param.ndim >= 2, "All parameters must have at least 2 dimensions."
        param_groups = []
        if world_size == 1:
            # For single GPU case, we don't need the update buffer
            param_groups.append(dict(params=params))
        else:
            for size in {p.numel() for p in params}:
                b = torch.empty(world_size, size, dtype=params[0].dtype, device=params[0].device)
                group = dict(params=[p for p in params if p.numel() == size],
                             update_buffer=b, 
                             update_buffer_views=[b[i] for i in range(world_size)])
                param_groups.append(group)
        
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params=param_groups, defaults=defaults)

        self._step = self._1_gpu_step if world_size == 1 else self._n_gpu_step

    @torch.no_grad()
    def step(self, closure=None):
        self._step()

    def _1_gpu_step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf: torch.tensor = state["momentum_buffer"]
                buf.lerp_(g, 1 - group["momentum"])
                g = g.lerp_(buf, group["momentum"]) if group["nesterov"] else buf
                g = self.zeropower_via_newtonschulz5(g, steps=group["ns_steps"])
                p.add_(g.view_as(p), alpha=-group["lr"] * max(1, p.size(-2) / p.size(-1))**0.5)
        return

    def _n_gpu_step(self):
        for group in self.param_groups:
            update_buffer: torch.tensor = group["update_buffer"]
            update_buffer_views: list[torch.tensor] = group["update_buffer_views"]
            # generate weight updates in distributed fashion
            params: list[torch.tensor] = group["params"]
            handle = None
            params_world = None
            def update_prev():
                handle.wait()
                for p_world, g_world in zip(params_world, update_buffer_views):
                    p_world.add_(g_world.view_as(p_world),
                                  alpha=-group["lr"] * max(1, p_world.size(-2) / p_world.size(-1))**0.5)
            for base_i in range(len(params))[::self.world_size]:
                if base_i + self.rank < len(params):
                    p = params[base_i + self.rank]
                    g = p.grad
                    assert g is not None
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf: torch.tensor = state["momentum_buffer"]
                    buf.lerp_(g, 1 - group["momentum"])
                    g = g.lerp_(buf, group["momentum"]) if group["nesterov"] else buf
                    g = self.zeropower_via_newtonschulz5(g, steps=group["ns_steps"]).flatten()
                else:
                    g = update_buffer_views[self.rank]
                if base_i > 0:
                    update_prev()
                handle = dist.all_gather_into_tensor(update_buffer, g, async_op=True)
                params_world = params[base_i : base_i + self.world_size]
            update_prev()

    def zeropower_via_newtonschulz5(self, G: torch.tensor, steps: int) -> torch.tensor:
        """
        Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
        quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
        of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
        zero even beyond the point where the iteration no longer converges all the way to one everywhere
        on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
        where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
        performance at all relative to UV^T, where USV^T = G is the SVD.
        """
        # batched Muon implementation by @scottjmaddox, and put into practice in the record by @YouJiacheng
        a, b, c = (3.4445, -4.7750,  2.0315)
        X = G.bfloat16()
        if G.size(-2) > G.size(-1):
            X = X.mT

        # Ensure spectral norm is at most 1
        X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
        # Perform the NS iterations
        for _ in range(steps):
            A = X @ X.mT
            B = b * A + c * A @ A # quintic computation strategy adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
            X = a * X + B @ X
        
        if G.size(-2) > G.size(-1):
            X = X.mT
        return X


def muon_param_filter(param: nn.Parameter) -> bool:
    """Filter for parameters suitable for Muon (2D or higher dimensional)"""
    # Device filter - uncomment when using CUDA
    #if 'cuda' not in str(param.device) and 'cpu' not in str(param.device):
    #    return False
    return param.ndim >= 2


__default_config__ = OptimizerConfig(
    optimizer_kwargs={
        'lr': 0.02,
        'momentum': 0.95, 
        'nesterov': True,
        'ns_steps': 5
    },
    param_filter=muon_param_filter,
    fallback_optimizer_class=torch.optim.AdamW,
    fallback_kwargs={'lr': 1e-3}
)