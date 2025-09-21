from gpt_lab.catalog_utils import SkipModuleException
from .nano_gpt import NanoGPT

__all__ = [
    "NanoGPT",
]

try:
    from .modded_nanogpt import ModdedNanoGPT
    __all__.append(
        "ModdedNanoGPT"
        )
except SkipModuleException:
    pass