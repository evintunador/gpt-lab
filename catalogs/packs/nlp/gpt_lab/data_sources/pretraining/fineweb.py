from typing import Callable, List, Optional, Union, Iterable
from pathlib import Path
from enum import Enum
import random

import torch
from torch.utils.data import Dataset, IterableDataset
from datasets import load_dataset

from gpt_lab.data_sources.catalog_utils import BinaryShardIO, SequentialPretokenizedDatasetMixin, Split


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
    """Map-style FineWeb dataset."""
    def __init__(
        self,
        size: FineWebSize = FineWebSize.v350B,
        edu: bool = False,
        seed: Optional[int] = None,
        world_size: int = 1,
        rank: int = 0,
        tokenizer_enc_func: Optional[Callable] = None,
        max_seq_len: Optional[int] = None,
        data_file_url: Optional[str] = None,
    ):
        if data_file_url:
            fw = load_dataset(
                "parquet",
                data_files={'train': data_file_url},
                split='train',
                cache_dir='./data/.cache/huggingface_fw',
            )
        else:
            fw = load_dataset(
                "HuggingFaceFW/fineweb" + ("-edu" if edu else ""),
                name="sample-" + size.value,
                split='train',
                streaming=False,
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
        data = self.data[i]["text"]
        if not self.tokenizer_enc_func:
            return data
        tokens = self.tokenizer_enc_func(data)
        if self.max_seq_len:
            tokens = tokens[:self.max_seq_len]
        return tokens


class FineWebStreamingDataset(IterableDataset):
    """Iterable-style FineWeb dataset."""
    def __init__(
        self,
        size: FineWebSize = FineWebSize.v350B,
        edu: bool = False,
        seed: Optional[int] = None,
        world_size: int = 1,
        rank: int = 0,
        tokenizer_enc_func: Optional[Callable] = None,
        max_seq_len: Optional[int] = None,
    ):
        fw = load_dataset(
            "HuggingFaceFW/fineweb" + ("-edu" if edu else ""),
            name="sample-" + size.value,
            split='train',
            streaming=True,
            cache_dir='./data/.cache/huggingface_fw',
        )
        self.data = fw.shuffle(seed=seed or random.randint(0, 2**32 - 1))
        self.tokenizer_enc_func = tokenizer_enc_func
        self.max_seq_len = max_seq_len

        if world_size > 1:
            self.data = self.data.shard(num_shards=world_size, index=rank)

    def __iter__(self) -> Iterable[dict]:
        worker_info = torch.utils.data.get_worker_info()
        iter_data = self.data
        if worker_info is not None:
            iter_data = self.data.shard(num_shards=worker_info.num_workers, index=worker_info.id)

        for rec in iter_data:
            data = rec["text"]
            if not self.tokenizer_enc_func:
                yield data
            else:
                tokens = self.tokenizer_enc_func(data)
                if self.max_seq_len:
                    tokens = tokens[:self.max_seq_len]
                yield tokens


def create_fineweb_dataset(
    streaming: bool = True,
    size: FineWebSize = FineWebSize.v350B,
    edu: bool = False,
    seed: Optional[int] = None,
    world_size: int = 1,
    rank: int = 0,
    tokenizer_enc_func: Optional[Callable] = None,
    max_seq_len: Optional[int] = None,
    data_file_url: Optional[str] = None,
):
    """Factory function to create either a map-style or iterable-style FineWeb dataset."""
    common_args = {
        "size": size,
        "edu": edu,
        "seed": seed,
        "world_size": world_size,
        "rank": rank,
        "tokenizer_enc_func": tokenizer_enc_func,
        "max_seq_len": max_seq_len,
    }
    if streaming:
        return FineWebStreamingDataset(**common_args)
    
    return FineWebDataset(data_file_url=data_file_url, **common_args)


class PrecachedFineWebDataset(SequentialPretokenizedDatasetMixin, Dataset):
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

        SequentialPretokenizedDatasetMixin.__init__(
            self,
            save_dir=save_dir,
            shard_size=shard_size,
            max_num_shards=max_num_shards,
            cache_filename_prefix=cache_filename_prefix,
            num_workers=num_workers,
        )

        token_dtype = BinaryShardIO.pick_token_dtype(vocab_size)

        if not self.has_cache():
            raw = FineWebStreamingDataset(
                size=size,
                edu=edu,
                seed=seed,
            )
            self.build_cache(
                doc_iter=iter(raw),
                tokenizer_encode_fn=tokenizer_encode_fn,
                doc_separator=doc_separator,
                token_dtype=token_dtype,
            )

        self.setup_cache_index(split.value, seq_len)
