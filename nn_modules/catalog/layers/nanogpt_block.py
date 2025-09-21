import torch.nn as nn

from nn_modules.catalog.sequence_mixing import CausalSelfAttention
from nn_modules.catalog.channel_mixing import MLP


class NanoGPTBlock(nn.Module):
    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float, bias: bool):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd, bias=bias)
        self.attn = CausalSelfAttention(n_embd=n_embd, n_head=n_head, block_size=block_size, dropout=dropout, bias=bias)
        self.ln_2 = nn.LayerNorm(n_embd, bias=bias)
        # unlike NanoGPT, our MLP does not support bias on th elinear layers -_-
        self.mlp = MLP(in_dim=n_embd, out_dim=n_embd, hidden_dim=4*n_embd, activation="gelu", dropout=dropout)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
