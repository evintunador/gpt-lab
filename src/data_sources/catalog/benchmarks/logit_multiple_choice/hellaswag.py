from typing import Dict, Any, Optional, Callable
import json
import os
import requests

from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset
from datasets import load_dataset

from data_sources.catalog_utils import Split
from src.benchmarks.protocols import LogitMultipleChoiceItem


"""
Example HellaSwag json item:

{
    "ind": 24, 
    "activity_label": "Roof shingle removal", 
    "ctx_a": "A man is sitting on a roof.", 
    "ctx_b": "he", 
    "ctx": "A man is sitting on a roof. he", 
    "split": "val", 
    "split_type": "indomain", 
    "label": 3, 
    "endings": [
        "is using wrap to wrap a pair of skis.", 
        "is ripping level tiles off.", 
        "is holding a rubik's cube.", 
        "starts pulling up roofing on a roof."
    ], 
    "source_id": "activitynet~v_-JhWjGDPHMY"
}

ind: dataset ID
activity_label: The ActivityNet or WikiHow label for this example
context: There are two formats. The full context is in ctx. When the context ends in an (incomplete) noun phrase, like for ActivityNet, this incomplete noun phrase is in ctx_b, and the context up until then is in ctx_a. This can be useful for models such as BERT that need the last sentence to be complete. However, it's never required. If ctx_b is nonempty, then ctx is the same thing as ctx_a, followed by a space, then ctx_b.
endings: a list of 4 endings. The correct index is given by label (0,1,2, or 3)
split: train, val, or test.
split_type: indomain if the activity label is seen during training, else zeroshot
source_id: Which video or WikiHow article this example came from

gpt2 (124M)
- eleuther harness reports acc 28.92%, acc_norm 31.14% (multiple choice style)

gpt2-xl (1558M)
- eleuther harness reports acc 40.04%, acc_norm 50.89% (multiple choice style)

The validation set of HellaSwag has a total of 10,042 examples.
"""


def download_file(url: str, fname: str, chunk_size=1024):
    """Helper function to download a file from a given url"""
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get("content-length", 0))
    with open(fname, "wb") as file, tqdm(
        desc=fname,
        total=total,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            bar.update(size)


class HellaSwagDataset(Dataset):
    """
    HellaSwag dataset that yields standardized LogitMultipleChoiceItem objects.
    
    Supports both streaming and downloading modes with flexible caching.
    """
    
    def __init__(
        self,
        split: Split = Split.VAL,
        cache_dir: Optional[str] = None,
        streaming: bool = False,
        limit: Optional[int] = None,
    ):
        """Initialize HellaSwag dataset.
        
        Args:
            split: Which split to use (train/val/test)
            cache_dir: Directory to cache downloaded data. If None, uses default cache.
            streaming: Whether to stream data instead of downloading
            limit: Maximum number of examples to load (useful for debugging)
        """
        self.split = split
        self.cache_dir = cache_dir
        self.streaming = streaming
        self.limit = min(int(limit), 1024) if limit is not None else limit
        
        # Set up cache path
        if cache_dir is None:
            cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "hellaswag")
        self.cache_path = os.path.join(cache_dir, f"hellaswag_{split.value}.jsonl")
        
        # Check if we have cached data for non-streaming mode
        if not streaming and self._has_cached_data():
            self._load_from_cache()
            return
        
        # Load dataset using HuggingFace datasets
        try:
            self.dataset = load_dataset(
                "Rowan/hellaswag",
                split=split.value,
                cache_dir=cache_dir,
                streaming=streaming,
            )
            
            # Convert to list if not streaming and apply limit
            if streaming:
                self.data = self.dataset if limit is None else self.dataset.take(limit)
            else:
                self.data = list(self.dataset) if limit is None else list(self.dataset.take(limit))
                if cache_dir:
                    self._save_to_cache()
                    
        except Exception as e:
            # Fallback to manual download if HuggingFace fails
            print(f"HuggingFace datasets failed: {e}")
            print("Falling back to manual download...")
            self._manual_download()
    
    def _manual_download(self):
        """Fallback method to manually download HellaSwag data."""
        # Use the existing download logic from above
        if self.cache_dir is None:
            self.cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "hellaswag")
        
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # URLs for different splits
        urls = {
            Split.TRAIN: "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_train.jsonl",
            Split.VAL: "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl", 
            Split.TEST: "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_test.jsonl"
        }
        
        data_url = urls[self.split]
        data_filename = os.path.join(self.cache_dir, f"hellaswag_{self.split.value}.jsonl")
        
        if not os.path.exists(data_filename):
            print(f"Downloading {data_url} to {data_filename}...")
            download_file(data_url, data_filename)
        
        # Load data from file
        self.data = []
        with open(data_filename, "r") as f:
            for i, line in enumerate(f):
                if self.limit is not None and i >= self.limit:
                    break
                example = json.loads(line)
                self.data.append(example)
    
    def __len__(self) -> int:
        """Return the length of the dataset."""
        if self.streaming:
            # For streaming datasets, we can't know the length without iterating
            return float('inf') if self.limit is None else self.limit
        return len(self.data)
    
    def __getitem__(self, idx: int) -> LogitMultipleChoiceItem:
        """Get a single example from the dataset."""
        if self.streaming:
            # For streaming, we need to handle indexing differently
            for i, example in enumerate(self.data):
                if i == idx:
                    return LogitMultipleChoiceItem(
                        context=example["ctx"],
                        choices=example["endings"],
                        label=example["label"],
                    )
            raise IndexError(f"Index {idx} out of range")
        
        example = self.data[idx]
        return LogitMultipleChoiceItem(
            context=example["ctx"],
            choices=example["endings"], 
            label=example["label"],
        )
    
    def _save_to_cache(self):
        """Save the current dataset to a cache file."""
        if self.streaming:
            raise ValueError("Cannot save streaming dataset to cache")
        
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        with open(self.cache_path, 'w') as f:
            for example in self.data:
                f.write(json.dumps(example) + '\n')
    
    def _has_cached_data(self) -> bool:
        """Check if cache file exists."""
        return os.path.exists(self.cache_path)
    
    def _load_from_cache(self):
        """Load dataset from a cache file."""
        if not self._has_cached_data():
            raise FileNotFoundError(f"Cache file not found: {self.cache_path}")
        
        self.data = []
        with open(self.cache_path, 'r') as f:
            for i, line in enumerate(f):
                if self.limit is not None and i >= self.limit:
                    break
                example = json.loads(line.strip())
                self.data.append(example)
        
        self.streaming = False


def render_hellaswag_example(example, tokenizer_encode_fn):
    """
    Given the example as a dictionary, render it as three torch tensors:
    - tokens (the tokens of context + completion, of size 4xN, as there are always 4 candidates)
    - mask (is 1 in the region of the candidate completion, where we evaluate likelihoods)
    - label (the index of the correct completion, which we hope has the highest likelihood)
    """
    ctx = example["ctx"]
    label = example["label"]
    endings = example["endings"]

    # gather up all the tokens
    ctx_tokens = tokenizer_encode_fn(ctx)
    tok_rows = []
    mask_rows = []
    for end in endings:
        end_tokens = tokenizer_encode_fn(" " + end)  # NOTE: prepending " " assuming GPT-2 based tokenizer
        tok_rows.append(ctx_tokens + end_tokens)
        mask_rows.append([0]*len(ctx_tokens) + [1]*len(end_tokens))

    # have to be careful during the collation because the number of tokens in each row can differ
    max_len = max(len(row) for row in tok_rows)
    tokens = torch.zeros((4, max_len), dtype=torch.int32)
    mask = torch.zeros((4, max_len), dtype=torch.int32)
    for i, (tok_row, mask_row) in enumerate(zip(tok_rows, mask_rows)):
        tokens[i, :len(tok_row)] = torch.tensor(tok_row)
        mask[i, :len(mask_row)] = torch.tensor(mask_row)

    return tokens, mask, label


class TokenizedHellaSwagDataset(HellaSwagDataset):
    def __init__(
            self, 
            tokenizer_encode_fn: Callable[[str], np.ndarray],
            **kwargs
        ):
        super().__init__(**kwargs)
        self.tokenizer_encode_fn = tokenizer_encode_fn

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        return render_hellaswag_example(
            super().__getitem__(idx), 
            self.tokenizer_encode_fn
        )