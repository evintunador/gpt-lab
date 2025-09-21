from gpt_lab.catalog_utils import SkipModuleException
from .nanogpt_block import NanoGPTBlock

__all__ = [
    "NanoGPTBlock",
]

# cuda-specific modules
try:
    from .modded_nanogpt_block import ModdedNanoGPTBlock
    __all__.append("ModdedNanoGPTBlock")
except SkipModuleException:
    pass