from typing import Callable, List, Optional, Union, Iterable
from pathlib import Path
from enum import Enum

from torch.utils.data import Dataset
from datasets import load_dataset
import random

from gpt_lab.data_sources.catalog_utils import Split, PrecachedDatasetMixin


"""
FineWeb dataset
https://huggingface.co/datasets/HuggingFaceFW/fineweb

example doc to highlight the structure of the dataset:
{
  "text": "Posted by mattsmith on 20th April 2012\nStraight from...",
  "id": "<urn:uuid:d853d453-196e-4488-a411-efc2b26c40d2>",
  "dump": "CC-MAIN-2013-20",
  "url": "http://nleastchatter.com/philliesphandom/tag/freddy-galvis/",
  "date": "2013-05-18T07:24:47Z",
  "file_path": "s3://commoncrawl/long.../path.../file.gz",
  "language": "en",
  "language_score": 0.9185474514961243,
  "token_count": 594
}
"""


class FineWebSize(Enum):
    v10B = "10BT"
    v100B = "100BT"
    v350B = "350BT"


class FineWebDataset(Dataset):
    def __init__(
        self, 
        size: FineWebSize = FineWebSize.v350B,
        edu: bool = False,
        streaming: bool = True,
        seed: Optional[int] = None,
        world_size: int = 1,
        rank: int = 0,
        tokenizer_enc_func: Callable = None,
        max_seq_len: int = None,
    ):
        self.streaming = streaming
        fw = load_dataset(
            "HuggingFaceFW/fineweb" + ("-edu" if edu else ""), 
            name="sample-" + size.value, 
            split='train', 
            streaming=streaming,
            cache_dir='./data/.cache/huggingface_fw',
        )
        self.data = fw.shuffle(seed=seed or random.randint(0, 2**32 - 1))
        self.tokenizer_enc_func = tokenizer_enc_func
        self.max_seq_len = max_seq_len

        if world_size > 1:
            self.data = self.data.shard(num_shards=world_size, index=rank)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i: int):
        if self.streaming:
            raise TypeError("Indexing not supported when streaming=True. Iterate instead.")

        data = self.data[i]["text"]
        
        if not self.tokenizer_enc_func:
            return data
            
        tokens = self.tokenizer_enc_func(data)
        
        if self.max_seq_len:
            tokens = tokens[:self.max_seq_len]
            
        return tokens

    def __iter__(self) -> Iterable[dict]:
        for rec in self.data:
            data = rec["text"]
            
            if not self.tokenizer_enc_func:
                yield data
            else:
                tokens = self.tokenizer_enc_func(data)
                
                if self.max_seq_len:
                    tokens = tokens[:self.max_seq_len]
                    
                yield tokens


class PrecachedFineWebDataset(Dataset, PrecachedDatasetMixin):
    def __init__(
        self, 
        save_dir: Union[str, Path],
        tokenizer_encode_fn: Callable[[str], List[int]],
        vocab_size: int,
        doc_separator: Optional[int] = None,
        seq_len: int = 2048,
        size: FineWebSize = FineWebSize.v350B,
        edu: bool = False,
        split: Split = Split.TRAIN,
        shard_size: int = 2**27,
        max_num_shards: Optional[int] = None,
        num_workers: Optional[int] = None,
        seed: Optional[int] = None,
    ):
        cache_filename_prefix = "finewebedu" if edu else "fineweb"

        PrecachedDatasetMixin.__init__(
            self,
            save_dir=save_dir,
            shard_size=shard_size,
            max_num_shards=max_num_shards,
            cache_filename_prefix=cache_filename_prefix,
            num_workers=num_workers,
        )

        token_dtype = PrecachedDatasetMixin.pick_token_dtype(vocab_size)

        if not self.has_cache():
            raw = FineWebDataset(
                size=size,
                edu=edu,
                split=Split.TRAIN,
                streaming=True,
                seed=seed,
            )
            self.build_cache(
                doc_iter=iter(raw),
                tokenizer_encode_fn=tokenizer_encode_fn,
                doc_separator=doc_separator,
                token_dtype=token_dtype,
            )

        self.setup_cache_index(split.value, seq_len)
