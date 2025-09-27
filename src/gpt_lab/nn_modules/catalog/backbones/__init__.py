import torch
from .nano_gpt import NanoGPTBackbone

__all__ = [
    "NanoGPTBackbone",
]

# Import CUDA-specific modules if CUDA is available
if torch.cuda.is_available():
    from .modded_nanogpt import ModdedNanoGPT
    __all__.append("ModdedNanoGPT")