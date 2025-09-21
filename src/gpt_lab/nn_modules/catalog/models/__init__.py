import torch
from .nano_gpt import NanoGPT

__all__ = [
    "NanoGPT",
]

# Import CUDA-specific modules if CUDA is available
if torch.cuda.is_available():
    from .modded_nanogpt import ModdedNanoGPT
    __all__.append("ModdedNanoGPT")