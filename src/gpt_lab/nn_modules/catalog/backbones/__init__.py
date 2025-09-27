import torch
from .nano_gpt import NanoGPTBackbone

__all__ = [
    "NanoGPTBackbone",
]

# Import CUDA-specific modules if CUDA is available
if torch.cuda.is_available():
    from .modded_nano_gpt import ModdedNanoGPTBackbone
    __all__.append("ModdedNanoGPTBackbone")