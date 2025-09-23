from typing import List, Tuple
import torch
import torch.nn.functional as F
import tiktoken

from gpt_lab.nn_modules.catalog.models.nano_gpt import NanoGPT
from gpt_lab.benchmarks import register_handler
from gpt_lab.benchmarks.catalog import MultipleChoiceItem, MultipleChoiceBenchmark, FillInTheBlankItem


class NanoGPTModel:
    """
    A high-level wrapper for the NanoGPT nn.Module that adds
    functionality for generation and benchmarking.
    """
    def __init__(self, nn_module: NanoGPT, tokenizer: tiktoken.Encoding):
        self.nn_module = nn_module
        self.tokenizer = tokenizer

    def to(self, device_or_dtype):
        self.nn_module.to(device_or_dtype)
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
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        """
        self.nn_module.eval()
        input_ids = self.tokenizer.encode(prompt)
        idx = torch.tensor(input_ids, dtype=torch.long, device=next(self.nn_module.parameters()).device)[None, ...]

        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.nn_module.block_size else idx[:, -self.nn_module.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self.nn_module(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
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
        self.nn_module.eval()
        results = []

        for item in batch:
            # 1. Generate the predicted answer string (using greedy decoding)
            predicted_full = self.generate(item.prompt, max_new_tokens=20, temperature=0.0)
            predicted_answer = predicted_full[len(item.prompt):]

            # 2. Calculate the Negative Log-Likelihood of the true answer
            prompt_tokens = self.tokenizer.encode(item.prompt)
            answer_tokens = self.tokenizer.encode(item.answer)
            if not answer_tokens:
                results.append((predicted_answer, 0.0))
                continue

            full_tokens = torch.tensor(prompt_tokens + answer_tokens, dtype=torch.long, device=next(self.nn_module.parameters()).device)
            
            # Get logits for the answer part
            logits, _ = self.nn_module(full_tokens[:-1].unsqueeze(0))
            answer_logits = logits[0, len(prompt_tokens)-1:, :]
            answer_targets = full_tokens[len(prompt_tokens):]
            
            loss = F.cross_entropy(answer_logits, answer_targets, reduction='sum')
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
        self.nn_module.eval()
        predictions = []

        for item in batch:
            # Render the example into tokens, mask, and label for each choice
            tokens, mask, _ = MultipleChoiceBenchmark.render_example(item, self.tokenizer.encode)
            tokens = tokens.to(next(self.nn_module.parameters()).device)
            mask = mask.to(next(self.nn_module.parameters()).device)
            
            completion_losses = []
            for i in range(tokens.shape[0]): # For each choice
                # The nn.Module's forward pass returns (logits, loss)
                _, loss = self.nn_module(tokens[i].unsqueeze(0), targets=tokens[i].unsqueeze(0))
                
                # We need to calculate loss only over the completion part
                # Re-calculate loss with masking
                logits, _ = self.nn_module(tokens[i, :-1].unsqueeze(0))
                targets = tokens[i, 1:]
                
                loss_per_token = F.cross_entropy(
                    logits.view(-1, logits.size(-1)), 
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
