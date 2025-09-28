import torch
from .nano_gpt import NanoGPTTrainingModel\

__all__ = [
    "NanoGPTTrainingModel",\
]

# Import CUDA-specific modules if CUDA is available
if torch.cuda.is_available():
    from .modded_nano_gpt import ModdedNanoGPTTrainingModel
    __all__.append("ModdedNanoGPTTrainingModel")