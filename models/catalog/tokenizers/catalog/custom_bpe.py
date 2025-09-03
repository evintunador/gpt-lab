import os
import argparse
import pickle
from itertools import chain
from typing import List, Dict

import regex
import tiktoken
import numpy as np
import random
from datasets import load_dataset
from tqdm import tqdm
import torch
import torch.distributed as dist

from data_sources.catalog.fineweb import FineWebDataset
from models.catalog.tokenizers.visualise import visualise_tokens


assert torch.cuda.is_available()
# Check if environment variables are set by torchrun, otherwise default to single GPU
if "RANK" in os.environ and "WORLD_SIZE" in os.environ and "LOCAL_RANK" in os.environ:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
else:
    rank = 0
    world_size = 1
    local_rank = 0
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size) 
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
device = torch.device("cuda", local_rank)
torch.cuda.set_device(device)
if world_size > 1:
    dist.init_process_group(backend="nccl", device_id=device)
    dist.barrier()
master_process = (rank == 0)


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


def prepare_data_tensors(n: int, dtype: torch.dtype, separator_flag: int, seed: int = random.randint(0, 2**32)):
    dataset = FineWebDataset(streaming=True, seed=seed)
    if world_size > 1:
        dataset = dist.DistributedSampler(dataset)
    data_buffer = []
    data_tensor = torch.tensor([], dtype=dtype, device=device)
    ranks = {}
    for i in range(2**8):
        ranks[bytes([i])] = i
    tot_ct = 0
    i = 0
    while tot_ct < n:
        try:
            doc = next(dataset)
        except Exception:
            break
        data_buffer.append(doc)
        tot_ct += len(doc)
        # some cheap vast.ai nodes have less CPU ram than GPU ram so we pipeline
        if len(data_buffer) > 2**16:
            data_buffer_bytes: List[List[bytes]] = [
                [bytes([b]) for b in wordish.encode("utf-8")]
                for wordish in regex.findall(pat_str, data_buffer)
            ]
            data_buffer_ids: List[List[int]] = [
                [nat2int(ranks[b]) for b in wordish]
                for wordish in data_buffer_bytes
            ]
            ids = torch.tensor(
                list(chain.from_iterable(word + [separator_flag] for word in data_buffer_ids)), 
                dtype=dtype, device=device)
            data_tensor = torch.concat((data_tensor, ids), dim=0)
            data_buffer = []
        i += 1
    return data_tensor, ranks


def pair_up(ids: torch.tensor):
    pairs = torch.stack((ids[:-1], ids[1:]), dim=0) # (2, words_in_data * (avg_word_len + 1))
    unique, counts = torch.unique(pairs, return_counts=True, dim=1)
        # shapes (2, very_long) and (very_long)
        # where very_long < words_in_data * (avg_word_len + 1)
    
    # use separator token between words to ensure we follow regex
    valid_mask = torch.all(unique != separator_flag, dim=0) # (very_long)
    unique = unique[:, valid_mask] # (2, very_long)
    counts = counts[valid_mask] # (very_long)

    return pairs, unique, counts


def multi_gpu_best_pair(counts: torch.tensor, unique: torch.tensor, k: int):
    # select top k pairs to go into consideration
    counts, sort_idx = torch.sort(counts, descending=True) # (very_long) and (very_long)
    pairs_idx = sort_idx[:k] # shape (k)
    most_common_pairs_local = unique[:, pairs_idx] # (2, k)
    counts_local = counts[:k]# (k)
    
    # communicate between GPUs
    most_common_pairs_global = torch.zeros((2, k * world_size), dtype=torch.float32, device=device)
    counts_global = torch.zeros(k * world_size, dtype=torch.float32, device=device)
    most_common_pairs_global[:, rank * k : (rank + 1) * k] = most_common_pairs_local.to(torch.float32)
    counts_global[rank * k : (rank + 1) * k] = counts_local.to(torch.float32)
    dist.all_reduce(most_common_pairs_global, op=dist.ReduceOp.SUM)
    dist.all_reduce(counts_global, op=dist.ReduceOp.SUM)

    # Count occurrences of each unique pair
    unique_pairs, inverse_indices = torch.unique(most_common_pairs_global.t(), dim=0, return_inverse=True)
    sum_counts = torch.zeros(unique_pairs.size(0), dtype=torch.float, device=device)
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


def merge(pairs, best_pair, removal_flag: int, new_token_id: int):
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
        k: int = 256
    ) -> Dict[bytes, int]:
    if master_process:
        demo_text = (f"This is a test of our custom trained BPE tokenizer on FineWeb data.\n"
                    f"It should handle punctuation, numbers (like 42 and 3.14159), and special characters ($#@!) properly.\n"
                    f"Supercalifragilisticexpialidocious antidisestablishmentarianism!!!")
        demo_words = [[bytes([b]) for b in word.encode("utf-8")] for word in regex.findall(pat_str, demo_text)]

    removal_flag = 32_767 if dtype == torch.int16 else 2_147_483_647

    progress_bar = tqdm(total=vocab_size - 256, unit="merges") if master_process else None
    for j in range(256, vocab_size):
        pairs, unique, counts = pair_up(ids=ids)

        if world_size > 1:
            best_pair = multi_gpu_best_pair()
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

        ids = merge(pairs=pairs, best_pair=best_pair, removal_flag=removal_flag, new_token_id=new_token_id)

        demo_words = slow_merge(demo_words, tuple(best_bytes), token_bytes)
        if j % 1000 == 0 or j in (256, vocab_size - 1):
            print(f"\nThe most common pair {int2nat(best_pair[0])} + {int2nat(best_pair[1])} "
                    f"which makes '{token_bytes}' our {len(ranks)}th token")
            flat_demo_tokens = [token for word in demo_words for token in word]
            visualise_tokens(flat_demo_tokens)

        if master_process:
            progress_bar.update(1)
    
    print(f"peak memory reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB")

    return ranks


def save_tokenizer(enc, name, vocab_size, sample_size):
    os.makedirs('tokenizers', exist_ok=True)
    full_filename = f"tokenizers/trained/{name}_v{vocab_size}_n{sample_size}.pkl"
    with open(__file__, 'r') as f:
        script_content = f.read()
    tokenizer_data = {
        "pat_str": enc.pat_str,
        "mergeable_ranks": enc.mergeable_ranks,
        "script_content": script_content  # Add the script content for backup
    }
    with open(full_filename, 'wb') as f:
        pickle.dump(tokenizer_data, f)
    print(f"Tokenizer saved to {full_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a custom BPE tokenizer")
    parser.add_argument("-n", "--samples_per_gpu", type=int, default=2**27, 
        help="Maximum number of text characters to use on each GPU during training (default 2^27 should fit on single GPU with 8gb of VRAM)")
    parser.add_argument("-v", "--vocab_size", type=int, default=2**16-2, 
        help="Size of the vocabulary to train (default 2**16-2)")
    parser.add_argument("-f", "--name", type=str, default="gpt4regex", 
        help="Filename prefix to save the tokenizer (default 'gpt4regex')")
    parser.add_argument("-k", type=int, default=256,
        help="number of top-k unique pairs set to be communicated between GPUs. set heuristically to 256")
    parser.add_argument("-p", "--pat_str", type=str, default="gpt5",
        help="Pattern string. Defaults to 'gpt5'. Options are {gpt2, gpt4, gpt5}")
    parser.add_argument("-s", "--seed", type=int, default=random.randint(0, 2**32))
    args = parser.parse_args()

    dtype = torch.int16 if args.vocab_size <= 2**16-2 else torch.int32
    separator_flag = -32_768 if dtype == torch.int16 else -2_147_483_648

    data, ranks = prepare_data_tensors(n=args.samples_per_gpu, dtype=dtype, separator_flag=separator_flag)

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
    pat_str = pat_str_dict[args.pat_str] if args.pat_str in pat_str_dict.keys() else args.pat_str

    mergeable_ranks = bpe_train(
        ids=data, 
        ranks=ranks, 
        vocab_size=args.vocab_size, 
        separator_flag=separator_flag, 
        k=args.k
    )

    if master_process:
        enc = tiktoken.Encoding(
            name=args.name,
            pat_str=pat_str,
            mergeable_ranks=mergeable_ranks,
            special_tokens={"<|endoftext|>": args.vocab_size}
        )
        test_str = f"hello world"
        assert enc.decode(enc.encode(test_str)) == test_str
        
        save_tokenizer(
            mergeable_ranks, 
            pat_str, 
            args.name, 
            args.vocab_size, 
            args.samples_per_gpu,
        )