from gpt_lab.catalog_utils import SkipModuleException

# Import modules that do not have hardware dependencies
from .causal_self_attention import CausalSelfAttention

__all__ = [
    "CausalSelfAttention",
]

# Attempt to import CUDA-specific modules
try:
    from .flex_self_attention import HalfTruncatedRotary, FlexSelfAttention
    from .modded_nanogpt_flex_self_attention import ModdedNanoGPTFlexSelfAttention
    
    __all__.extend([
        "HalfTruncatedRotary",
        "FlexSelfAttention",
        "ModdedNanoGPTFlexSelfAttention",
    ])
except SkipModuleException:
    pass
