from torch.optim import AdamW
from optimizers.bulk_test_bench_utils import OptimizerConfig, OptimizerBenchmarkConfig

__default_config__ = OptimizerConfig(
    optimizer_kwargs={
        'lr': 1e-3,
        'betas': (0.9, 0.999),
        'eps': 1e-8,
        'weight_decay': 1e-2
    }
)

# Benchmark configuration with parameter sweeps
__benchmark_config__ = OptimizerBenchmarkConfig(
    optimizer_name='AdamW',
    competitors={
        'AdamW': {'class': AdamW}
    },
    parameter_space={
        'lr': [1e-1, 1e-2, 1e-3, 1e-4],
        'beta1': [0.8, 0.9, 0.95],
        'beta2': [0.99, 0.999, 0.9999], 
        'weight_decay': [1e-1, 1e-2]
    },
    optimizer_kwargs_builder=lambda params: {
        'lr': params['lr'],
        'betas': (params['beta1'], params['beta2']),
        'eps': 1e-8,
        'weight_decay': params['weight_decay']
    }
)
