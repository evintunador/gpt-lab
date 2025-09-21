from gpt_lab.catalog_utils import SkipModuleException
from .nano_gpt import NanoGPTModel

__all__ = [
    "NanoGPTModel",
]

# cuda-specific modules
try:
    from .modded_nano_gpt import ModdedNanoGPT
    __all__.append("ModdedNanoGPT")
except SkipModuleException:
    pass