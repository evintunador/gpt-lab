import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torch.nn.attention.flex_attention import create_block_mask

from modules.catalog.layers.modded_nanogpt_block import ModdedNanoGPTBlock
from modules.catalog.fp8_linear import FP8Linear
from modules.catalog.norms.rms_norm import RMSNorm
from modules.catalog.utils import next_multiple



class GPT(nn.Module):
    def __init__(self, 
    vocab_size: int, num_layers: int, num_val_emb: int, num_heads: int, model_dim: int, max_seq_len: int, mlp_ratio: int
    ):
        super().__init__()
        self.model_dim = model_dim
        self.max_seq_len = max_seq_len
        self.embed = nn.Embedding(vocab_size, model_dim)
        # token value embeddings by @KoszarskyB - inspired by @Grad62304977's value residual implementation following https://arxiv.org/abs/2410.17897
        # value embedding code simplification inspired by @ragulpr https://github.com/KellerJordan/modded-nanogpt/pull/78
        self.value_embeds = nn.ModuleList([nn.Embedding(vocab_size, model_dim) for _ in range(num_val_emb)])
        self.blocks = nn.ModuleList([ModdedNanoGPTBlock(model_dim, num_heads, mlp_ratio, max_seq_len, i) for i in range(num_layers)])
        # there are only 50257 unique GPT-2 tokens; we extend to nearest multiple of 128 for efficiency.
        # suggested by @Grad62304977. this originates from Karpathy's experiments.
        self.lm_head = FP8Linear(model_dim, next_multiple(vocab_size, n=128),
                                    use_fp8=False, x_s=(model_dim**0.5)/448, w_s=24/448, grad_s=1/448)
        self.lm_head.weight.detach().zero_() # @Grad62304977
        # Add learnable skip connection weights for decoder layers
        self.skip_weights = nn.Parameter(torch.ones(num_layers//2))
        self.norm = RMSNorm()

    def forward(self, input_seq: Tensor, target_seq: Tensor = None):
        assert input_seq.ndim == 1 # shape (B*N)

        # value emeddings provide extra info about a token at the first & final few layers
        ve = [value_embed(input_seq) for value_embed in self.value_embeds] # each (B*N, D)
        ve = [ve[i] for i in range(len(ve))] + [None] * (len(self.blocks) - len(ve)*2) + [ve[i] for i in range(len(ve))]
        assert len(ve) == len(self.blocks)

        # creating flex-attentio mask
        docs = (input_seq == 50256).cumsum(0)
        def doc_causal(b, h, q_idx, kv_idx):
            causal_mask = q_idx >= kv_idx
            document_mask = docs[q_idx] == docs[kv_idx]
            return causal_mask & document_mask
        # Because the sparsity pattern is independent of batch and heads, we'll set them to None (which broadcasts them) 
        block_mask = create_block_mask(doc_causal, B=None, H=None, Q_LEN=len(input_seq), KV_LEN=len(input_seq))

        x = x0 = self.norm(self.embed(input_seq)[None]) # use of norm here by @Grad62304977

        # U-net design by @brendanh0gan
        skip_connections = []
        n = len(self.skip_weights)
        for i in range(len(self.blocks)):
            if i >= n:
                x = x + self.skip_weights[i - n] * skip_connections.pop()
            x = self.blocks[i](x, ve[i], x0, block_mask)
            if i < n:
                skip_connections.append(x)

        x = self.norm(x)
        logits = self.lm_head(x).float()
        # @Grad62304977 added tanh softcapping following Gemma 2 paper, @KoszarskyB reduced it from 30 to 15, @YouJiacheng shifted it by +15 (2*sigmoid(2*x)=tanh(x)+1)
        logits = 30 * torch.sigmoid(logits / (7.5 * x.size(-1)**0.5))

        if target_seq is None:
            return logits
        else:
            return F.cross_entropy(logits.view(-1, logits.size(-1)), target_seq, 
                                  reduction='sum' if self.training else 'mean')

    def get_num_params(self):
        """
        Return the number of parameters in the model.
        """
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        assert idx.ndim == 1
        def cdiv(m, n):
            return (m + (n - 1)) // n
        seq_len = idx.size(0)
        if seq_len % 128 != 0:
            pad_ct = cdiv(seq_len, 128) * 128 - seq_len
            idx = torch.cat((idx, torch.zeros(pad_ct, dtype=idx.dtype, device=idx.device)), dim=0)
        
        self.eval()  # Ensure model is in evaluation mode
        for _ in range(max_new_tokens):
            # Forward pass to get logits
            logits = self(idx[-self.max_seq_len:] if idx.size(0) > self.max_seq_len else idx)
            # Focus on the last token's prediction
            logits = logits[0, min(seq_len, self.max_seq_len) - 1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[-1]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx[min(seq_len, self.max_seq_len)] = idx_next

            # iterate sequence count and account for any time we surpass flex-attention's block size
            seq_len += 1
            if (seq_len - 1) % 128 == 0:
                pad_ct = cdiv(seq_len, 128) * 128 - seq_len
                idx = torch.cat((idx, [0] * pad_ct), dim=0)

        return idx[:seq_len]