from typing import List, Tuple
import torch
import torch.nn.functional as F
import tiktoken

from gpt_lab.nn_modules.backbone import NanoGPTBackbone
from gpt_lab.benchmarks import register_handler
from gpt_lab.benchmarks.multiple_choice import MultipleChoiceItem, MultipleChoiceBenchmark
from gpt_lab.benchmarks.fill_in_the_blank import FillInTheBlankItem


class NanoGPTModel:
    """
    A high-level wrapper for the NanoGPT nn.Module that adds
    functionality for generation and benchmarking.
    """
    def __init__(self, backbone: NanoGPTBackbone, tokenizer: tiktoken.Encoding):
        self.backbone = backbone
        self.tokenizer = tokenizer

        self.max_seq_len = backbone.max_seq_len

    def to(self, device_or_dtype):
        self.backbone.to(device_or_dtype)
        return self

    def eval(self):
        self.backbone.eval()
        return self
    
    def train(self):
        self.backbone.train()
        return self

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: int, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        """
        self.backbone.eval()
        input_ids = self.tokenizer.encode(prompt)
        idx = torch.tensor(input_ids, dtype=torch.long, device=next(self.backbone.parameters()).device)[None, ...]

        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.backbone.max_seq_len else idx[:, -self.backbone.max_seq_len:]
            # forward the model to get the logits for the index in the sequence
            logits = self.backbone(idx_cond) # (B, T, V)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :]
            
            # handle temperature-based sampling
            if temperature > 0.0:
                logits = logits / temperature
                # optionally crop the logits to only the top k options
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                # apply softmax to convert logits to (normalized) probabilities
                probs = F.softmax(logits, dim=-1)
                # sample from the distribution
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                # greedy decoding
                _, idx_next = torch.topk(logits, k=1, dim=-1)

            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return self.tokenizer.decode(idx[0].tolist())

    @register_handler(benchmark_type="fill_in_the_blank")
    @torch.no_grad()
    def benchmark_fill_in_the_blank(self, batch: List[FillInTheBlankItem]) -> List[Tuple[str, float]]:
        """
        Handles fill-in-the-blank benchmarks.
        For each item, it generates a predicted answer and calculates the NLL of the true answer.
        """
        self.backbone.eval()
        results = []

        for item in batch:
            # 1. Generate the predicted answer string (using greedy decoding)
            # Use temperature 0 for greedy decoding to get deterministic output
            predicted_full = self.generate(item.prompt, max_new_tokens=20, temperature=0.0) 
            predicted_answer = predicted_full[len(item.prompt):]

            # 2. Calculate the Negative Log-Likelihood of the true answer
            prompt_tokens = self.tokenizer.encode(item.prompt)
            answer_tokens = self.tokenizer.encode(item.answer)
            if not answer_tokens:
                results.append((predicted_answer, 0.0))
                continue

            full_tokens = torch.tensor(prompt_tokens + answer_tokens, dtype=torch.long, device=next(self.backbone.parameters()).device)
            
            # Get logits for the answer part
            logits = self.backbone(full_tokens[:-1].unsqueeze(0))
            answer_logits = logits[0, len(prompt_tokens)-1:, :]
            answer_targets = full_tokens[len(prompt_tokens):]
            
            loss = F.cross_entropy(answer_logits.view(-1, answer_logits.shape[-1]), answer_targets.view(-1), reduction='sum')
            nll = loss.item()
            
            results.append((predicted_answer, nll))

        return results

    @register_handler(benchmark_type="multiple_choice")
    @torch.no_grad()
    def benchmark_multiple_choice(self, batch: List[MultipleChoiceItem]) -> List[int]:
        """
        Handles multiple-choice benchmarks by calculating the perplexity of each completion
        and choosing the one with the lowest loss.
        """
        self.backbone.eval()
        predictions = []

        for item in batch:
            # Render the example into tokens, mask, and label for each choice
            tokens, mask, _ = MultipleChoiceBenchmark.render_example(item, self.tokenizer.encode)
            tokens = tokens.to(next(self.backbone.parameters()).device)
            mask = mask.to(next(self.backbone.parameters()).device)
            
            completion_losses = []
            for i in range(tokens.shape[0]): # For each choice
                input_tokens = tokens[i, :-1].unsqueeze(0)
                targets = tokens[i, 1:].unsqueeze(0)
                
                # We need to calculate loss only over the completion part
                # Re-calculate loss with masking
                logits = self.backbone(input_tokens)
                
                loss_per_token = F.cross_entropy(
                    logits.view(-1, logits.shape[-1]), 
                    targets.view(-1), 
                    reduction='none'
                )
                
                # Apply mask (mask corresponds to completion tokens)
                masked_loss = loss_per_token * mask[i, 1:]
                
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
