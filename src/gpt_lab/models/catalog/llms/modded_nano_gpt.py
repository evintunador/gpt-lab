from typing import List, Tuple
import torch
import torch.nn.functional as F
import tiktoken

from gpt_lab.nn_modules.catalog_utils import ignore_if_no_cuda

# Check for CUDA availability before importing CUDA-specific modules
ignore_if_no_cuda()

from gpt_lab.nn_modules.catalog.models.modded_nanogpt import ModdedNanoGPT
from gpt_lab.benchmarks import register_handler
from gpt_lab.benchmarks.catalog import MultipleChoiceItem, MultipleChoiceBenchmark, FillInTheBlankItem


class ModdedNanoGPTModel:
    """
    A high-level wrapper for the ModdedNanoGPT nn.Module that adds
    functionality for generation and benchmarking.
    """
    def __init__(self, nn_module: ModdedNanoGPT, tokenizer: tiktoken.Encoding):
        self.nn_module = nn_module
        self.tokenizer = tokenizer

    def to(self, device):
        self.nn_module.to(device)
        return self

    def eval(self):
        self.nn_module.eval()
        return self
    
    def train(self):
        self.nn_module.train()
        return self

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: int, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        """
        self.nn_module.eval()
        input_ids = self.tokenizer.encode(prompt)
        idx = torch.tensor(input_ids, dtype=torch.int32, device=self.nn_module.embed.weight.device)

        assert idx.ndim == 1
        def cdiv(m, n):
            return (m + (n - 1)) // n
        seq_len = idx.size(0)
        
        # Pad initial sequence to be a multiple of 128 for flex attention
        if seq_len % 128 != 0:
            pad_ct = cdiv(seq_len, 128) * 128 - seq_len
            idx = torch.cat((idx, torch.zeros(pad_ct, dtype=idx.dtype, device=idx.device)), dim=0)

        for _ in range(max_new_tokens):
            # Forward pass to get logits, ensuring we don't exceed max_seq_len
            input_chunk = idx[-self.nn_module.max_seq_len:] if idx.size(0) > self.nn_module.max_seq_len else idx
            logits = self.nn_module(input_chunk)

            # Focus on the last token's prediction
            focus_idx = min(seq_len, self.nn_module.max_seq_len) - 1
            logits = logits[0, focus_idx, :] / temperature
            
            # Optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[-1]] = -float('Inf')
            
            # Apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            
            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append sampled index to the running sequence
            idx[min(seq_len, self.nn_module.max_seq_len)] = idx_next

            # Iterate sequence count and re-pad if we cross a block boundary
            seq_len += 1
            if (seq_len - 1) % 128 == 0:
                pad_ct = cdiv(seq_len, 128) * 128 - seq_len
                pad_tensor = torch.zeros(pad_ct, dtype=idx.dtype, device=idx.device)
                idx = torch.cat((idx, pad_tensor), dim=0)

        return self.tokenizer.decode(idx[:seq_len].tolist())

    @register_handler(benchmark_type="fill_in_the_blank")
    @torch.no_grad()
    def benchmark_fill_in_the_blank(self, batch: List[FillInTheBlankItem]) -> List[Tuple[str, float]]:
        """
        Handles fill-in-the-blank benchmarks.
        For each item, it generates a predicted answer and calculates the NLL of the true answer.
        """
        self.nn_module.eval()
        results = []

        for item in batch:
            # 1. Generate the predicted answer string
            predicted_full = self.generate(item.prompt, max_new_tokens=10, temperature=0.0) # Greedy
            predicted_answer = predicted_full[len(item.prompt):]

            # 2. Calculate the Negative Log-Likelihood of the true answer
            prompt_tokens = self.tokenizer.encode(item.prompt)
            answer_tokens = self.tokenizer.encode(item.answer)
            if not answer_tokens: # Handle empty answer strings
                results.append((predicted_answer, 0.0))
                continue

            full_tokens = torch.tensor(prompt_tokens + answer_tokens, dtype=torch.int32, device=self.nn_module.embed.weight.device)
            logits = self.nn_module(full_tokens, target_seq=None)
            
            answer_logits = logits[:, len(prompt_tokens)-1:-1, :]
            answer_targets = torch.tensor(answer_tokens, dtype=torch.int64, device=self.nn_module.embed.weight.device)
            
            loss = F.cross_entropy(
                answer_logits.view(-1, answer_logits.size(-1)),
                answer_targets.view(-1),
                reduction='sum'
            )
            nll = loss.item()
            
            results.append((predicted_answer, nll))

        return results

    @register_handler(benchmark_type="multiple_choice")
    @torch.no_grad()
    def benchmark_multiple_choice(self, batch: List[MultipleChoiceItem]) -> List[int]:
        """
        Handles multiple-choice benchmarks for a generative model by calculating
        the perplexity of each completion and choosing the most likely one.
        """
        self.nn_module.eval()
        predictions = []

        for item in batch:
            # Render the example into tokens, mask, and label
            tokens, mask, label = MultipleChoiceBenchmark.render_example(item, self.tokenizer.encode)
            tokens = tokens.to(self.nn_module.embed.weight.device)
            mask = mask.to(self.nn_module.embed.weight.device)
            
            completion_losses = []
            for i in range(tokens.shape[0]): # For each of the 4 choices
                # Get the logits from the model's forward pass
                # The nn.Module's forward pass is designed for training and returns loss
                # So we call it with targets=None to get logits
                logits = self.nn_module(tokens[i], target_seq=None)
                
                # We only want the logits for the completion part
                # Slice to targets (remove first token) and apply mask
                completion_logits = logits[:, :-1, :]
                completion_targets = tokens[i, 1:]
                
                # Calculate loss only for the completion tokens
                loss = F.cross_entropy(
                    completion_logits.view(-1, completion_logits.size(-1)), 
                    completion_targets.view(-1), 
                    reduction='none'
                )
                
                # Reshape loss and apply mask
                loss = loss.view(completion_targets.shape)
                masked_loss = loss * mask[i, 1:] # Mask corresponds to completion
                
                # Sum the loss for the completion and normalize by length
                # to get something proportional to perplexity
                num_completion_tokens = mask[i, 1:].sum()
                if num_completion_tokens > 0:
                    avg_loss = masked_loss.sum() / num_completion_tokens
                    completion_losses.append(avg_loss.item())
                else:
                    completion_losses.append(float('inf'))
            
            # The prediction is the choice with the minimum average loss
            prediction = completion_losses.index(min(completion_losses))
            predictions.append(prediction)

        return predictions
