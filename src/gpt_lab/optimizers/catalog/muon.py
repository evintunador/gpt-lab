from typing import Type, Tuple
import torch.distributed as dist
try:
    from muon import MuonWithAuxAdam, SingleDeviceMuonWithAuxAdam
except ImportError:
    # If muon package is not available, we'll define placeholder classes
    # This allows the module to be imported without error during discovery
    class MuonWithAuxAdam:
        def __init__(self, *args, **kwargs):
            raise ImportError("Muon package not installed. Install with: pip install -e '.[optimizers]'")
    
    class SingleDeviceMuonWithAuxAdam:
        def __init__(self, *args, **kwargs):
            raise ImportError("Muon package not installed. Install with: pip install -e '.[optimizers]'")

from gpt_lab.optimizers.catalog_utils import OptimizerConfig, OptimizerBenchmarkConfig


"""
To apply Muon to your model:
hidden_weights = [p for p in model.body.parameters() if p.ndim >= 2]
It should only be applied to 2d hidden parameters
hidden_gains_biases = [p for p in model.body.parameters() if p.ndim < 2]
nonhidden_params = [*model.head.parameters(), *model.embed.parameters()]
param_groups = [
    dict(params=hidden_weights, use_muon=True,
         lr=0.02, weight_decay=0.01),
    dict(params=hidden_gains_biases+nonhidden_params, use_muon=False,
         lr=3e-4, betas=(0.9, 0.95), weight_decay=0.01),
]
optimizer = MuonWithAuxAdam(param_groups)
"""


class Muon:
    """Adapter to make MuonWithAuxAdam's input parameter style fit with my own test/bench system"""
    def __new__(cls, *args, **kwargs):
        # Choose the appropriate base class based on distributed availability
        if dist.is_available() and dist.is_initialized():
            base_class = MuonWithAuxAdam
        else:
            base_class = SingleDeviceMuonWithAuxAdam
        
        # Create a new class that inherits from the appropriate base
        class MuonAdapter(base_class):
            def __init__(
                self, 
                params, 
                muon_lr: float,
                adamw_lr: float, 
                adamw_betas: Tuple[int, int] = (0.9, 0.95),
                momentum: float = 0.95, 
                nesterov: bool = True, 
                ns_steps: int = 5,
                weight_decay = 0.01,
            ):
                # NOTE: Muon really also shouldn't be used with input embeddings/convolutions & output linear layer
                params_2d = [p for p in params if p.ndim >= 2]
                params_1d = [p for p in params if p.ndim == 1]
                param_groups = [
                    dict(params=params_2d, use_muon=True, lr=muon_lr, momentum=momentum, weight_decay=weight_decay),
                    dict(params=params_1d, use_muon=False, lr=adamw_lr, betas=adamw_betas, weight_decay=weight_decay)
                ]
                super().__init__(param_groups)
        
        return MuonAdapter(*args, **kwargs)


__test_config__ = OptimizerConfig(
    optimizer_kwargs={
        'muon_lr': 0.02,
        'adamw_lr': 3e-4,
        'adamw_betas': (0.9, 0.95),
        'momentum': 0.95, 
        'nesterov': True,
        'ns_steps': 5,
        'weight_decay': 0.01,
    },
)


__benchmark_config__ = OptimizerBenchmarkConfig(
    optimizer_name = 'Muon',
    competitors = {'Muon': {'class': Muon}},
    parameter_space = {
        'muon_lr': [1e-1, 1e-2, 1e-3, 1e-4],
        'momentum': [0.9, 0.99, 0.999],
        'ns_steps': [3, 5, 7]
    },
    optimizer_kwargs_builder = lambda params: {
        'lr': params['lr'],
        'momentum': params['momentum'],
        'ns_steps': params['ns_steps']
    },
)