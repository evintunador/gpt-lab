import torch

# Import modules that do not have hardware dependencies
from .causal_self_attention import CausalSelfAttention

__all__ = [
    "CausalSelfAttention",
]

# Import CUDA-specific modules if CUDA is available
if torch.cuda.is_available():
    from .flex_self_attention import HalfTruncatedRotary, FlexSelfAttention
    from .modded_nanogpt_flex_self_attention import ModdedNanoGPTFlexSelfAttention
    
    __all__.extend([
        "HalfTruncatedRotary",
        "FlexSelfAttention",
        "ModdedNanoGPTFlexSelfAttention",
    ])
