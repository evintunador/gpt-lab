import torch
import torch.nn as nn
from torch import Tensor

from modules.catalog.sequence_mixing.modded_nanogpt_flex_self_attention import ModdedNanoGPTFlexSelfAttention, BlockMask
from modules.catalog.channel_mixing.mlp import MLP
from modules.catalog.norms.rms_norm import RMSNorm


class ModdedNanoGPTBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: int, max_seq_len: int, layer_idx: int):
        super().__init__()
        # skip attention of blocks.7 (the 8th layer) by @YouJiacheng
        # Adjusted for smaller models - only skip if we have enough layers
        skip_attn = (layer_idx == 7) and (dim > 512)  # Only skip in larger models
        self.attn = ModdedNanoGPTFlexSelfAttention(dim, num_heads, max_seq_len) if not skip_attn else None
        self.mlp = MLP(dim, mlp_ratio)
        self.lambdas = nn.Parameter(torch.tensor([1., 0.]))
        self.norm = RMSNorm()

    def forward(self, x: Tensor, ve: Tensor | None, x0: Tensor, block_mask: BlockMask):
        x = self.lambdas[0] * x + self.lambdas[1] * x0
        if self.attn is not None:
            x = x + self.attn(self.norm(x), ve, block_mask)
        x = x + self.mlp(self.norm(x))
        return x