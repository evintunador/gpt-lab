import torch
from .nanogpt_block import NanoGPTBlock

__all__ = [
    "NanoGPTBlock",
]

# Import CUDA-specific modules if CUDA is available
if torch.cuda.is_available():
    from .modded_nanogpt_block import ModdedNanoGPTBlock
    __all__.append("ModdedNanoGPTBlock")