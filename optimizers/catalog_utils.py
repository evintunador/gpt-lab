from typing import Dict, Any, Optional, Callable, Type, Union, Tuple, List, Set
from dataclasses import dataclass, field
import warnings

import torch
import torch.nn as nn


@dataclass
class OptimizerConfig:
    """Configuration for optimizer testing and benchmarking"""
    optimizer_kwargs: Dict[str, Any] = field(default_factory=dict)
    param_filter: Optional[Callable[[nn.Parameter], bool]] = None
    fallback_optimizer_class: Type[torch.optim.Optimizer] = torch.optim.AdamW
    fallback_kwargs: Dict[str, Any] = field(default_factory=lambda: {'lr': 1e-3})


@dataclass
class OptimizerBenchmarkConfig:
    """Configuration for optimizer benchmarking (similar to BenchmarkConfig for modules)"""
    # A friendly name for the optimizer, used for filenaming
    optimizer_name: str
    # A dictionary mapping competitor names to their optimizer classes
    competitors: Dict[str, Dict[str, Any]]  # {'Adam': {'class': Adam}, 'AdamW': {'class': AdamW}}
    # The parameter space to sweep
    parameter_space: Dict[str, List[Any]]
    # A function that takes a parameter combination and returns optimizer_kwargs
    optimizer_kwargs_builder: Callable[[Dict[str, Any]], Dict[str, Any]]
    # Optional parameter filter for mixed optimizers
    param_filter: Optional[Callable[[nn.Parameter], bool]] = None
    # Fallback configuration
    fallback_optimizer_class: Type[torch.optim.Optimizer] = torch.optim.AdamW
    fallback_kwargs: Dict[str, Any] = field(default_factory=lambda: {'lr': 1e-3})

    def __post_init__(self):
        self.optimizer_name = self.optimizer_name.replace('/', '_').replace('\\', '_')


@dataclass
class ParameterConstraints:
    """Discovered constraints for an optimizer"""
    min_ndim: Optional[int] = None
    max_ndim: Optional[int] = None
    min_numel: Optional[int] = None
    max_numel: Optional[int] = None
    allowed_dtypes: Optional[Set[torch.dtype]] = None
    disallowed_dtypes: Optional[Set[torch.dtype]] = None
    allowed_devices: Optional[Set[str]] = None
    disallowed_devices: Optional[Set[str]] = None
    
    def to_filter(self) -> Callable[[nn.Parameter], bool]:
        """Convert constraints to a parameter filter function"""
        def filter_fn(param: nn.Parameter) -> bool:
            # Check ndim constraints
            if self.min_ndim is not None and param.ndim < self.min_ndim:
                return False
            if self.max_ndim is not None and param.ndim > self.max_ndim:
                return False
            
            # Check numel constraints
            if self.min_numel is not None and param.numel() < self.min_numel:
                return False
            if self.max_numel is not None and param.numel() > self.max_numel:
                return False
            
            # Check dtype constraints
            if self.allowed_dtypes is not None and param.dtype not in self.allowed_dtypes:
                return False
            if self.disallowed_dtypes is not None and param.dtype in self.disallowed_dtypes:
                return False
            
            # Check device constraints
            device_str = str(param.device)
            if self.allowed_devices is not None and device_str not in self.allowed_devices:
                return False
            if self.disallowed_devices is not None and device_str in self.disallowed_devices:
                return False
            
            return True
        
        return filter_fn
    
    def __str__(self) -> str:
        constraints = []
        if self.min_ndim is not None:
            constraints.append(f"ndim >= {self.min_ndim}")
        if self.max_ndim is not None:
            constraints.append(f"ndim <= {self.max_ndim}")
        if self.min_numel is not None:
            constraints.append(f"numel >= {self.min_numel}")
        if self.max_numel is not None:
            constraints.append(f"numel <= {self.max_numel}")
        if self.allowed_dtypes is not None:
            constraints.append(f"dtype in {[str(dt) for dt in self.allowed_dtypes]}")
        if self.disallowed_dtypes is not None:
            constraints.append(f"dtype not in {[str(dt) for dt in self.disallowed_dtypes]}")
        if self.allowed_devices is not None:
            constraints.append(f"device in {list(self.allowed_devices)}")
        if self.disallowed_devices is not None:
            constraints.append(f"device not in {list(self.disallowed_devices)}")
        
        return " AND ".join(constraints) if constraints else "no constraints"


def create_test_parameters(device: str) -> List[nn.Parameter]:
    """Create a diverse set of test parameters with different properties"""
    test_params = []
    
    # Different dtypes to test - but only those supported on the device
    base_dtypes = [torch.float32, torch.float16, torch.bfloat16]
    if device != "mps":  # MPS doesn't support float64
        base_dtypes.append(torch.float64)
    
    # Test which dtypes actually work on this device
    dtypes = []
    for dtype in base_dtypes:
        try:
            torch.tensor(1.0, dtype=dtype, device=device)
            dtypes.append(dtype)
        except Exception:
            continue
    
    # Different shapes/dimensions to test
    shapes = [
        # 0D (scalar) - skip as many optimizers don't handle these well
        # 1D (bias-like)
        [10], [100], [1000],
        # 2D (linear weight-like)  
        [10, 10], [100, 50], [1000, 500],
        # 3D (conv1d-like)
        [32, 16, 3], [64, 32, 5],
        # 4D (conv2d-like)
        [32, 16, 3, 3], [64, 32, 5, 5],
        # 5D (conv3d-like)
        [16, 8, 3, 3, 3]
    ]
    
    for dtype in dtypes:
        for shape in shapes:
            try:
                # Check if tensor creation would exceed memory
                numel = 1
                for dim in shape:
                    numel *= dim
                if numel > 100_000_000:  # Skip very large tensors during discovery
                    continue
                    
                param = nn.Parameter(torch.randn(*shape, dtype=dtype, device=device))
                test_params.append(param)
            except Exception:
                # Some dtype/device/shape combinations might not work
                continue
    
    return test_params


def test_optimizer_with_params(
    optimizer_class: Type[torch.optim.Optimizer],
    params: List[nn.Parameter],
    config: OptimizerConfig
) -> bool:
    """Test if an optimizer can be created with the given parameters"""
    try:
        optimizer = optimizer_class(params, **config.optimizer_kwargs)
        # Try to perform one step to catch runtime errors
        dummy_loss = sum(p.sum() for p in params)
        dummy_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        return True
    except Exception:
        return False


def discover_parameter_constraints(
    optimizer_class: Type[torch.optim.Optimizer],
    config: OptimizerConfig,
    device: str = "cpu"
) -> ParameterConstraints:
    """
    Systematically discover parameter constraints for an optimizer.
    
    This function creates parameters with different properties and tests
    which combinations work, then infers the constraints.
    """
    
    test_params = create_test_parameters(device)
    
    # Test each parameter individually to find basic constraints
    working_params = []
    failing_params = []
    
    for param in test_params:
        if test_optimizer_with_params(optimizer_class, [param], config):
            working_params.append(param)
        else:
            failing_params.append(param)
    
    if not working_params:
        print(f"[WARNING] No individual parameters work for {optimizer_class.__name__}")
        return ParameterConstraints()
    
    if not failing_params:
        return ParameterConstraints()
    
    constraints = ParameterConstraints()
    
    # Analyze ndim constraints
    working_ndims = {p.ndim for p in working_params}
    failing_ndims = {p.ndim for p in failing_params}
    
    if failing_ndims - working_ndims:  # Some ndims always fail
        if all(ndim < min(working_ndims) for ndim in failing_ndims - working_ndims):
            constraints.min_ndim = min(working_ndims)
        if all(ndim > max(working_ndims) for ndim in failing_ndims - working_ndims):
            constraints.max_ndim = max(working_ndims)
    
    # Analyze numel constraints
    working_numels = {p.numel() for p in working_params}
    failing_numels = {p.numel() for p in failing_params}
    
    if failing_numels - working_numels:  # Some numels always fail
        if all(numel < min(working_numels) for numel in failing_numels - working_numels):
            constraints.min_numel = min(working_numels)
        if all(numel > max(working_numels) for numel in failing_numels - working_numels):
            constraints.max_numel = max(working_numels)
    
    # Analyze dtype constraints
    working_dtypes = {p.dtype for p in working_params}
    failing_dtypes = {p.dtype for p in failing_params}
    
    # If some dtypes never work, they're disallowed
    always_failing_dtypes = failing_dtypes - working_dtypes
    if always_failing_dtypes:
        constraints.disallowed_dtypes = always_failing_dtypes
    
    # If only specific dtypes work (and others fail), set allowed dtypes
    if failing_dtypes and len(working_dtypes) < len(working_dtypes | failing_dtypes):
        # Only set allowed_dtypes if it's restrictive
        constraints.allowed_dtypes = working_dtypes
    
    # Test mixed parameter scenarios to refine constraints
    if len(working_params) > 1:
        # Test if mixing different types of working parameters still works
        sample_combinations = [
            working_params[:2],  # First two working params
            working_params[-2:], # Last two working params
            [working_params[0], working_params[-1]] if len(working_params) > 2 else working_params[:2]
        ]
        
        for combo in sample_combinations:
            if not test_optimizer_with_params(optimizer_class, combo, config):
                # Mixed parameters fail - might need more specific constraints
                print(f"[INFO] Mixed parameters fail for {optimizer_class.__name__}, constraints may be more complex")
                break
    
    return constraints


def auto_detect_optimizer_constraints(
    optimizer_class: Type[torch.optim.Optimizer],
    model: nn.Module,
    config: OptimizerConfig
) -> Tuple[bool, Optional[str], Optional[ParameterConstraints]]:
    """
    Attempt to detect if an optimizer has parameter constraints.
    
    Returns:
        (success, error_message, discovered_constraints)
    """
    try:
        # Try to create the optimizer with all parameters
        optimizer = optimizer_class(model.parameters(), **config.optimizer_kwargs)
        return True, None, None
    except Exception as e:
        # If it fails, discover constraints
        device = next(model.parameters()).device
        constraints = discover_parameter_constraints(optimizer_class, config, str(device))
        return False, str(e), constraints


def create_smart_optimizer(
    model: nn.Module,
    optimizer_class: Type[torch.optim.Optimizer],
    config: Optional[OptimizerConfig] = None
) -> Union[torch.optim.Optimizer, 'MixedOptimizer']:
    """
    Smart optimizer creation with intelligent constraint discovery.
    """
    if config is None:
        config = OptimizerConfig()
    
    # Strategy 1: Try the optimizer as-is
    success, error, discovered_constraints = auto_detect_optimizer_constraints(optimizer_class, model, config)
    if success:
        return optimizer_class(model.parameters(), **config.optimizer_kwargs)
    
    # Strategy 2: If we have explicit param_filter, try mixed approach
    if config.param_filter is not None:
        try:
            return create_mixed_optimizer(model, optimizer_class, config)
        except Exception as mixed_error:
            warnings.warn(f"Explicit mixed optimizer approach failed for {optimizer_class.__name__}: {mixed_error}")
    
    # Strategy 3: Use discovered constraints to create mixed optimizer
    if discovered_constraints and discovered_constraints != ParameterConstraints():
        try:
            inferred_config = OptimizerConfig(
                optimizer_kwargs=config.optimizer_kwargs,
                param_filter=discovered_constraints.to_filter(),
                fallback_optimizer_class=config.fallback_optimizer_class,
                fallback_kwargs=config.fallback_kwargs
            )
            return create_mixed_optimizer(model, optimizer_class, inferred_config)
        except Exception as inferred_error:
            warnings.warn(f"Constraint-based approach failed for {optimizer_class.__name__}: {inferred_error}")
    
    # Strategy 4: Fall back to known-good optimizer
    warnings.warn(
        f"Could not create {optimizer_class.__name__} (error: {error}). "
        f"Falling back to {config.fallback_optimizer_class.__name__}."
    )
    return config.fallback_optimizer_class(model.parameters(), **config.fallback_kwargs)


def create_mixed_optimizer(
    model: nn.Module,
    primary_optimizer_class: Type[torch.optim.Optimizer],
    config: OptimizerConfig
) -> Union[torch.optim.Optimizer, 'MixedOptimizer']:
    """
    Creates a mixed optimizer that applies different optimizers to different parameters
    based on the parameter filter.
    """
    if config.param_filter is None:
        return primary_optimizer_class(model.parameters(), **config.optimizer_kwargs)
    
    # Split parameters based on filter
    primary_params = []
    fallback_params = []
    
    for param in model.parameters():
        if config.param_filter(param):
            primary_params.append(param)
        else:
            fallback_params.append(param)
    
    if not primary_params:
        raise ValueError(f"No parameters match filter for {primary_optimizer_class.__name__}")
    
    if not fallback_params:
        return primary_optimizer_class(model.parameters(), **config.optimizer_kwargs)
    
    # Create separate optimizers for each parameter group
    primary_optimizer = primary_optimizer_class(primary_params, **config.optimizer_kwargs)
    fallback_optimizer = config.fallback_optimizer_class(fallback_params, **config.fallback_kwargs)
    
    return MixedOptimizer(primary_optimizer, fallback_optimizer)


class MixedOptimizer:
    """A mixed optimizer that delegates to multiple optimizers"""
    
    def __init__(self, primary_optimizer, fallback_optimizer):
        self.primary_optimizer = primary_optimizer
        self.fallback_optimizer = fallback_optimizer
        
    def zero_grad(self, set_to_none=False):
        self.primary_optimizer.zero_grad(set_to_none=set_to_none)
        self.fallback_optimizer.zero_grad(set_to_none=set_to_none)
        
    def step(self, closure=None):
        self.primary_optimizer.step(closure)
        self.fallback_optimizer.step(closure)
        
    def state_dict(self):
        return {
            'primary': self.primary_optimizer.state_dict(),
            'fallback': self.fallback_optimizer.state_dict()
        }
        
    def load_state_dict(self, state_dict):
        self.primary_optimizer.load_state_dict(state_dict['primary'])
        self.fallback_optimizer.load_state_dict(state_dict['fallback'])


# Predefined parameter filters for common use cases
def large_param_filter(param: nn.Parameter) -> bool:
    """Filter for parameters with many elements"""
    return param.numel() > 100


def conv_param_filter(param: nn.Parameter) -> bool:
    """Filter for convolutional parameters (3D+ usually)"""
    return param.ndim >= 3


def linear_param_filter(param: nn.Parameter) -> bool:
    """Filter for linear layer parameters (2D weights)"""
    return param.ndim == 2


def bias_param_filter(param: nn.Parameter) -> bool:
    """Filter for bias parameters (1D)"""
    return param.ndim == 1
