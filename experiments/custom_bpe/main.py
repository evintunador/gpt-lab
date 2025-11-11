import os
import argparse
import pickle
import logging
from itertools import chain
from typing import List, Dict

import regex
import tiktoken
import numpy as np
import random
from tqdm import tqdm
import torch
import torch.distributed as dist

from gpt_lab.data_sources.pretraining import create_fineweb_dataset
from gpt_lab.distributed import DistributedManager
from gpt_lab.configuration import get_config
from gpt_lab.reproducibility import ReproducibilityManager, get_system_info
from gpt_lab.logger import setup_experiment_logging

logger = logging.getLogger(__name__)


"""
this script relies on operations that are not available on mps, 
and there'd be no point running it on CPU
"""
assert torch.cuda.is_available()


def visualise_tokens(token_values: list[bytes]) -> None:
    background = [f"\u001b[48;5;{i}m" for i in [167, 179, 185, 77, 80, 68, 134]]
    # If token boundaries do not occur at unicode character boundaries, it's unclear how best to
    # demo the token. Here, we'll just use the unicode replacement character to represent some
    # fraction of a character.
    unicode_token_values = [x.decode("utf-8", errors="replace") for x in token_values]

    running_length = 0
    last_color = None
    for token in unicode_token_values:
        color = background[running_length % len(background)]
        if color == last_color:
            color = background[(running_length + 1) % len(background)]
            assert color != last_color
        last_color = color
        running_length += len(token)
        print(color + token, end="")
    print("\u001b[0m")


def nat2int(num: int):
    """
    converts natural numbers to integer counterparts for use in
    efficiently utilizing signed int datatypes
    (0, 1, 2, 3,...) -> (0, -1, 1, -2, 2,...)
    unsigned dtypes would be preferable but pytorch lacks support
    for many key operations on unsigned dtypes
    """
    if num % 2 == 0:  # even numbers map to positive
        return num // 2
    else: # odd numbers map to negative
        return -(num + 1) // 2


def int2nat(num: int):
    """
    converts integer numbers to natural counterparts for use in
    efficiently utilizing signed int datatypes
    (0, -1, 1, -2, 2,...) -> (0, 1, 2, 3,...)
    unsigned dtypes would be preferable but pytorch lacks support
    for many key operations on unsigned dtypes
    """
    if num >= 0:  # positive numbers map back to even
        return 2 * num
    else:  # negative numbers map back to odd
        return -2 * num - 1


def slow_merge(words, most_common_pair, token_bytes):
    new_words = []
    for word in words:
        new_word = []
        i = 0
        while i < len(word) - 1:
            if (word[i], word[i + 1]) == most_common_pair:
                new_word.append(token_bytes)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        if i == len(word) - 1:
            new_word.append(word[i])
        new_words.append(new_word)
    return new_words


def prepare_data_tensors(
    n: int, 
    dtype: torch.dtype, 
    separator_flag: int, 
    dist_manager: DistributedManager, 
    pat_str: str,
    seed: int = random.randint(0, 2**32),
):
    dataset = create_fineweb_dataset(
        streaming=True, 
        seed=seed,
        world_size=dist_manager.world_size,
        rank=dist_manager.rank,
    )
    data_buffer = []
    data_tensor = torch.tensor([], dtype=dtype, device=dist_manager.device)
    ranks = {}
    for i in range(2**8):
        ranks[bytes([i])] = i
    tot_ct = 0
    i = 0
    data_iter = iter(dataset)
    while tot_ct < n:
        try:
            doc = next(data_iter)
        except StopIteration:
            break
        data_buffer.append(doc)
        tot_ct += len(doc)
        # some cheap vast.ai nodes have less CPU ram than GPU ram so we pipeline
        if len(data_buffer) > 2**16:
            text_chunk = "\n".join(data_buffer)
            data_buffer_bytes: List[List[bytes]] = [
                [bytes([b]) for b in wordish.encode("utf-8")]
                for wordish in regex.findall(pat_str, text_chunk)
            ]
            data_buffer_ids: List[List[int]] = [
                [nat2int(ranks[b]) for b in wordish]
                for wordish in data_buffer_bytes
            ]
            ids = torch.tensor(
                list(chain.from_iterable(word + [separator_flag] for word in data_buffer_ids)), 
                dtype=dtype, device=dist_manager.device)
            data_tensor = torch.concat((data_tensor, ids), dim=0)
            data_buffer = []
        i += 1

    if data_buffer:
        text_chunk = "\n".join(data_buffer)
        data_buffer_bytes: List[List[bytes]] = [
            [bytes([b]) for b in wordish.encode("utf-8")]
            for wordish in regex.findall(pat_str, text_chunk)
        ]
        data_buffer_ids: List[List[int]] = [
            [nat2int(ranks[b]) for b in wordish]
            for wordish in data_buffer_bytes
        ]
        ids = torch.tensor(
            list(chain.from_iterable(word + [separator_flag] for word in data_buffer_ids)), 
            dtype=dtype, device=dist_manager.device)
        data_tensor = torch.concat((data_tensor, ids), dim=0)

    return data_tensor, ranks


def pair_up(ids: torch.tensor, separator_flag: int):
    pairs = torch.stack((ids[:-1], ids[1:]), dim=0) # (2, words_in_data * (avg_word_len + 1))
    unique, counts = torch.unique(pairs, return_counts=True, dim=1)
        # shapes (2, very_long) and (very_long)
        # where very_long < words_in_data * (avg_word_len + 1)

    # use separator token between words to ensure we follow regex
    valid_mask = torch.all(unique != separator_flag, dim=0) # (very_long)
    unique = unique[:, valid_mask] # (2, very_long)
    counts = counts[valid_mask] # (very_long)

    return pairs, unique, counts


def multi_gpu_best_pair(counts: torch.tensor, unique: torch.tensor, k: int, dist_manager: DistributedManager):
    # select top k pairs to go into consideration
    counts, sort_idx = torch.sort(counts, descending=True) # (very_long) and (very_long)
    pairs_idx = sort_idx[:k] # shape (k)
    most_common_pairs_local = unique[:, pairs_idx] # (2, k)
    counts_local = counts[:k]# (k)
    # communicate between GPUs
    most_common_pairs_global = torch.zeros(
        (2, k * dist_manager.world_size), 
        dtype=torch.float32, 
        device=dist_manager.device
    )
    counts_global = torch.zeros(
        k * dist_manager.world_size, 
        dtype=torch.float32, 
        device=dist_manager.device
    )
    most_common_pairs_global[:, dist_manager.rank * k : (dist_manager.rank + 1) * k] = most_common_pairs_local.to(torch.float32)
    counts_global[dist_manager.rank * k : (dist_manager.rank + 1) * k] = counts_local.to(torch.float32)
    dist.all_reduce(most_common_pairs_global, op=dist.ReduceOp.SUM)
    dist.all_reduce(counts_global, op=dist.ReduceOp.SUM)

    # Count occurrences of each unique pair
    unique_pairs, inverse_indices = torch.unique(most_common_pairs_global.t(), dim=0, return_inverse=True)
    sum_counts = torch.zeros(unique_pairs.size(0), dtype=torch.float, device=dist_manager.device)
    sum_counts.scatter_add_(0, inverse_indices, counts_global.float())
    pair_occurrences = torch.bincount(inverse_indices)
    max_occurrence = torch.max(pair_occurrences)

    # Filter to only consider pairs with the maximum occurrence count
    max_occurrence_mask = (pair_occurrences == max_occurrence)
    filtered_sum_counts = sum_counts[max_occurrence_mask]
    filtered_unique_pairs = unique_pairs[max_occurrence_mask]

    # Find the pair with the largest count
    max_index = torch.argmax(filtered_sum_counts)
    best_pair = filtered_unique_pairs[max_index].cpu().numpy() # (2)
    return best_pair


def merge(ids, pairs, best_pair, removal_flag: int, new_token_id: int):
    pair_mask = (pairs[0] == best_pair[0]) & (pairs[1] == best_pair[1]) 
    ids[:-1][pair_mask] = nat2int(new_token_id)
    ids[1:][pair_mask] = removal_flag
    keep_mask = (ids != removal_flag)
    ids = ids[keep_mask]
    return ids


def bpe_train(
        ids: torch.tensor, 
        ranks: Dict[bytes, int], 
        vocab_size: int, 
        separator_flag: int,
        pat_str: str,
        dtype: torch.dtype,
        dist_manager: DistributedManager,
        k: int = 256,
    ) -> Dict[bytes, int]:
    if dist_manager.is_main_process:
        demo_text = (f"This is a test of our custom trained BPE tokenizer on FineWeb data.\n"
                    f"It should handle punctuation, numbers (like 42 and 3.14159), and special characters ($#@!) properly.\n"
                    f"Supercalifragilisticexpialidocious antidisestablishmentarianism!!!")
        demo_words = [[bytes([b]) for b in word.encode("utf-8")] for word in regex.findall(pat_str, demo_text)]

    removal_flag = 32_767 if dtype == torch.int16 else 2_147_483_647

    progress_bar = tqdm(total=vocab_size - 256, unit="merges") if dist_manager.is_main_process else None
    for j in range(256, vocab_size):
        pairs, unique, counts = pair_up(ids=ids, separator_flag=separator_flag)

        if dist_manager.world_size > 1:
            best_pair = multi_gpu_best_pair(counts, unique, k, dist_manager)
            if best_pair is None:
                logger.info("No more mergeable pairs found. Ending training early.")
                break
        else:
            pair_idx = torch.argmax(counts) # (1)
            best_pair = unique[:, pair_idx].cpu().numpy() # (2)

        # Map token IDs back to the corresponding byte sequences
        best_bytes = [None, None]
        best_pair_0 = int2nat(best_pair[0])
        best_pair_1 = int2nat(best_pair[1])
        for bytes_token, id_token in ranks.items():
            if id_token == best_pair_0:
                best_bytes[0] = bytes_token
            if id_token == best_pair_1:
                best_bytes[1] = bytes_token
        token_bytes = best_bytes[0] + best_bytes[1]
        new_token_id = len(ranks)
        # Add the new token!
        ranks[token_bytes] = new_token_id

        ids = merge(ids=ids, pairs=pairs, best_pair=best_pair, removal_flag=removal_flag, new_token_id=new_token_id)

        if dist_manager.is_main_process:
            demo_words = slow_merge(demo_words, tuple(best_bytes), token_bytes)
            if j % 1000 == 0 or j in (256, vocab_size - 1):
                logger.info(
                    f"\nThe most common pair {int2nat(best_pair[0])} + {int2nat(best_pair[1])} "
                    f"which makes '{token_bytes}' our {len(ranks)}th token"
                )
                flat_demo_tokens = [token for word in demo_words for token in word]
                visualise_tokens(flat_demo_tokens)

        if dist_manager.is_main_process:
            progress_bar.update(1)

    logger.info(f"peak memory reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB")

    return ranks


def save_tokenizer(output_dir: str, enc, name, vocab_size, sample_size):
    """Saves the tokenizer to the specified output directory."""
    full_filename = os.path.join(output_dir, f"{name}_v{vocab_size}_n{sample_size}.pkl")
    
    tokenizer_data = {
        "pat_str": enc._pat_str,
        "mergeable_ranks": enc._mergeable_ranks,
    }
    with open(full_filename, 'wb') as f:
        pickle.dump(tokenizer_data, f)
    print(f"Tokenizer saved to {full_filename}")


def run(config: dict, dist_manager: DistributedManager, repro_manager: ReproducibilityManager):
    if repro_manager.output_dir:
        setup_experiment_logging(
            log_dir=repro_manager.output_dir,
            rank=dist_manager.rank,
            is_main_process=dist_manager.is_main_process
        )
    logger.info("System Information", extra=get_system_info(git_info=repro_manager.get_git_info()))
    logger.info("Hyperparameters", extra=config)


    dtype = torch.int16 if config['vocab_size'] <= 2**16-2 else torch.int32
    separator_flag = -32_768 if dtype == torch.int16 else -2_147_483_648

    pat_str_dict = {
        "gpt2": (r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}++| ?\p{N}++| ?[^\s\p{L}\p{N}]++|\s++$|\s+(?!\S)|\s"""),
        "gpt4": (r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}+| ?[^\s\p{L}\p{N}]++[\r\n]*+|\s++$|\s*[\r\n]|\s+(?!\S)|\s"""),
        "gpt5": "|".join([
            r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
            r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
            r"""\p{N}{1,3}""",
            r""" ?[^\s\p{L}\p{N}]+[\r\n/]*""",
            r"""\s*[\r\n]+""",
            r"""\s+(?!\S)""",
            r"""\s+""",
        ]),
    }
    pat_str = pat_str_dict[config['pat_str']] if config['pat_str'] in pat_str_dict.keys() else config['pat_str']

    data, ranks = prepare_data_tensors(
        n=config['samples_per_gpu'], dtype=dtype, separator_flag=separator_flag, dist_manager=dist_manager, seed=config['seed'], pat_str=pat_str
    )

    mergeable_ranks = bpe_train(
        ids=data, 
        ranks=ranks, 
        vocab_size=config['vocab_size'], 
        separator_flag=separator_flag,
        pat_str=pat_str,
        dtype=dtype,
        dist_manager=dist_manager,
        k=config['k']
    )

    if dist_manager.is_main_process:
        enc = tiktoken.Encoding(
            name=config['tokenizer_name'],
            pat_str=pat_str,
            mergeable_ranks=mergeable_ranks,
            special_tokens={"<|endoftext|>": config['vocab_size']}
        )
        test_str = f"hello world"
        assert enc.decode(enc.encode(test_str)) == test_str

        if repro_manager.output_dir:
            peak_mem = torch.cuda.max_memory_reserved() // 1024 // 1024
            logger.info(f"Final peak memory reserved: {peak_mem} MiB", extra={"peak_mem_mib": peak_mem})

        save_tokenizer(
            output_dir=repro_manager.output_dir,
            enc=enc,
            name=config['tokenizer_name'],
            vocab_size=config['vocab_size'],
            sample_size=config['samples_per_gpu'] * dist_manager.world_size,
        )


if __name__ == "__main__":
    # --- Configuration ---
    parser = argparse.ArgumentParser(description="Train a custom BPE tokenizer")
    parser.add_argument("-n", "--samples-per-gpu", dest="samples_per_gpu", type=int,
        help="Maximum number of text characters to use on each GPU during training.")
    parser.add_argument("-v", "--vocab-size", dest="vocab_size", type=int,
        help="Size of the vocabulary to train.")
    parser.add_argument("-f", "--tokenizer-name", dest="tokenizer_name", type=str,
        help="Filename prefix to save the tokenizer.")
    parser.add_argument("-k", dest="k", type=int,
        help="Number of top-k unique pairs set to be communicated between GPUs.")
    parser.add_argument("-p", "--pat-str", dest="pat_str", type=str,
        help="Pattern string. Options are {gpt2, gpt4, gpt5} or a custom pattern.")
    parser.add_argument("-s", "--seed", dest="seed", type=int, help="Seed for the data loader.")

    with DistributedManager() as dist_manager:
        config = get_config(parser)
        runs_dir = os.path.join(os.path.dirname(__file__), "runs")
        with ReproducibilityManager(
            output_dir=runs_dir,
            is_main_process=dist_manager.is_main_process,
        ) as repro_manager:
            # Broadcast the output directory from the main process to all other processes
            repro_manager.output_dir = dist_manager.broadcast_object(repro_manager.output_dir)
            
            run(config, dist_manager, repro_manager)