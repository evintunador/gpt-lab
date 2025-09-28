import torch
from .nano_gpt import NanoGPTModel

__all__ = [
    "NanoGPTModel",
]

# Import CUDA-specific modules if CUDA is available
if torch.cuda.is_available():
    from .modded_nano_gpt import ModdedNanoGPTModel
    __all__.append("ModdedNanoGPTModel")