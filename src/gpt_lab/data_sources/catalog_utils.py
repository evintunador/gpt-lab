from enum import Enum
from typing import Optional, Union, Callable, Iterable, List
from pathlib import Path
import os
import multiprocessing as mp
from functools import partial
import requests

import torch
import numpy as np
from tqdm import tqdm


class Split(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


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


class PrecachedDatasetMixin:
    """
    Utility for building and reading token cache shards:
    - Shards are .bin files with 256*int32 header + tokens of uint{8,16,32,64}
    - First shard is 'val'; others are 'train'
    - Also provides indexing and slicing across shards given a seq_len
    """

    def __init__(
        self,
        save_dir: Union[str, Path],
        shard_size: int = 2**27,
        max_num_shards: Optional[int] = None,
        cache_filename_prefix: str = "dataset",
        num_workers: Optional[int] = None,
    ):
        self._cache_dir = Path(save_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._shard_size = int(shard_size)
        self._max_num_shards = max_num_shards
        self._cache_filename_prefix = cache_filename_prefix
        self._num_workers = num_workers or max(1, (os.cpu_count() or 2) - 2)

        # indexing-related state (populated by setup_cache_index)
        self._files: List[Path] = []
        self._memmaps: List[np.memmap] = []
        self._shard_sizes: Optional[np.ndarray] = None
        self._cumsum: Optional[np.ndarray] = None
        self._seq_len: Optional[int] = None
        self._num_items: int = 0
        self._split: Optional[str] = None

    @staticmethod
    def pick_token_dtype(vocab_size: int) -> np.dtype:
        """Selects the smallest uint dtype that can hold the vocabulary."""
        if vocab_size < 2**8:
            return np.uint8
        elif vocab_size < 2**16:
            return np.uint16
        elif vocab_size < 2**32:
            return np.uint32
        else:
            return np.uint64

    def _cache_glob(self, split: str) -> List[Path]:
        return sorted(self._cache_dir.glob(f"{self._cache_filename_prefix}_{split}_*.bin"))

    def has_cache(self) -> bool:
        return bool(self._cache_glob("val") and self._cache_glob("train"))

    def build_cache(
        self,
        doc_iter: Iterable[str],
        tokenizer_encode_fn: Callable[[str], List[int]],
        doc_separator: Optional[int] = None,
        token_dtype: np.dtype = np.uint16,
    ) -> None:
        if self.has_cache():
            return

        shard_index = 0
        token_count = 0
        all_tokens_np = np.empty((self._shard_size,), dtype=token_dtype)
        pbar = None

        use_workers = self._num_workers > 1
        worker = PrecachedDatasetMixin._tokenize_worker

        if use_workers:
            with mp.Pool(self._num_workers) as pool:
                imap_iter = pool.imap(
                    partial(worker, encode=tokenizer_encode_fn, doc_separator=doc_separator),
                    doc_iter,
                    chunksize=16,
                )
                for tokens in imap_iter:
                    if token_count + len(tokens) < self._shard_size:
                        all_tokens_np[token_count:token_count + len(tokens)] = tokens
                        token_count += len(tokens)
                        if pbar is None:
                            pbar = tqdm(total=self._shard_size, unit="tokens", desc=f"Shard {shard_index}")
                        pbar.update(len(tokens))
                    else:
                        split = "val" if shard_index == 0 else "train"
                        filename = self._cache_dir / f"{self._cache_filename_prefix}_{split}_{shard_index:06d}.bin"
                        remainder = self._shard_size - token_count
                        if pbar is None:
                            pbar = tqdm(total=self._shard_size, unit="tokens", desc=f"Shard {shard_index}")
                        pbar.update(remainder)
                        all_tokens_np[token_count:token_count + remainder] = tokens[:remainder]
                        PrecachedDatasetMixin.write_datafile(str(filename), all_tokens_np)
                        shard_index += 1

                        if self._max_num_shards is not None and shard_index >= self._max_num_shards + 1:
                            break

                        pbar = None
                        leftover = len(tokens) - remainder
                        all_tokens_np[0:leftover] = tokens[remainder:]
                        token_count = leftover
        else:
            for doc in doc_iter:
                tokens = worker(
                    doc,
                    encode=tokenizer_encode_fn,
                    doc_separator=doc_separator
                )
                if token_count + len(tokens) < self._shard_size:
                    all_tokens_np[token_count:token_count + len(tokens)] = tokens
                    token_count += len(tokens)
                    if pbar is None:
                        pbar = tqdm(total=self._shard_size, unit="tokens", desc=f"Shard {shard_index}")
                    pbar.update(len(tokens))
                else:
                    split = "val" if shard_index == 0 else "train"
                    filename = self._cache_dir / f"{self._cache_filename_prefix}_{split}_{shard_index:06d}.bin"
                    remainder = self._shard_size - token_count
                    if pbar is None:
                        pbar = tqdm(total=self._shard_size, unit="tokens", desc=f"Shard {shard_index}")
                    pbar.update(remainder)
                    all_tokens_np[token_count:token_count + remainder] = tokens[:remainder]
                    PrecachedDatasetMixin.write_datafile(str(filename), all_tokens_np)
                    shard_index += 1

                    if self._max_num_shards is not None and shard_index >= self._max_num_shards + 1:
                        break

                    pbar = None
                    leftover = len(tokens) - remainder
                    all_tokens_np[0:leftover] = tokens[remainder:]
                    token_count = leftover

        if token_count != 0 and (self._max_num_shards is None or shard_index < self._max_num_shards + 1):
            split = "val" if shard_index == 0 else "train"
            filename = self._cache_dir / f"{self._cache_filename_prefix}_{split}_{shard_index:06d}.bin"
            PrecachedDatasetMixin.write_datafile(str(filename), all_tokens_np[:token_count])

    def _split_to_str(self, split: Union[str, "Enum"]) -> str:
        return split.value if hasattr(split, "value") else str(split)
    
    def setup_cache_index(self, split: Union[str, "Enum"], seq_len: int) -> None:
        """
        Open memmaps and prepare sharded indexing for the given split and seq_len
        """
        split_str = self._split_to_str(split)
        self._split = split_str
        self._seq_len = int(seq_len)

        self._files = self._cache_glob(split_str)
        if not self._files:
            raise RuntimeError(f"No cached shards found for split='{split_str}' in {self._cache_dir}")
        self._memmaps = [PrecachedDatasetMixin.read_datafile_tokens_memmap(p) for p in self._files]
        self._shard_sizes = np.array([PrecachedDatasetMixin.read_datafile_token_count(p) for p in self._files], dtype=np.int64)
        self._cumsum = np.cumsum(np.concatenate([[0], self._shard_sizes]))
        total_tokens = int(self._shard_sizes.sum())
        self._num_items = total_tokens // self._seq_len

    def __len__(self) -> int:
        if self._seq_len is None:
            raise RuntimeError("Cache index not initialized. Call setup_cache_index(split, seq_len) first.")
        return self._num_items

    def __getitem__(self, idx: int) -> torch.Tensor:
        if self._seq_len is None:
            raise RuntimeError("Cache index not initialized. Call setup_cache_index(split, seq_len) first.")
        if idx < 0 or idx >= self._num_items:
            raise IndexError(idx)
        start = idx * self._seq_len
        end = start + self._seq_len
        return self._slice_tokens(start, end)

    def _slice_tokens(self, start: int, end: int) -> torch.Tensor:
        out = np.empty((end - start,), dtype=self._memmaps[0].dtype)
        write_pos = 0
        cur = start
        while cur < end:
            k = int(np.searchsorted(self._cumsum, cur, side="right")) - 1
            shard_offset = cur - int(self._cumsum[k])
            shard_available = int(self._shard_sizes[k] - shard_offset)
            need = end - cur
            take = min(shard_available, need)
            mm = self._memmaps[k]
            out[write_pos:write_pos + take] = mm[shard_offset:shard_offset + take]
            write_pos += take
            cur += take
        return torch.from_numpy(out)

    @staticmethod
    def write_datafile(filename, toks: np.ndarray):
        """ 
        Saves token data as a .bin file, for reading in C.
        - First comes a header with 256 int32s
        - The tokens follow
        """
        assert len(toks) < 2**31, "token count too large"  # ~2.1B tokens
        dtype_size = toks.dtype.itemsize
        header = np.zeros(256, dtype=np.int32)
        header[0] = 11041999   # magic number for file format identification/validation
        header[1] = 1          # version
        header[2] = len(toks)  # number of tokens after the header
        header[3] = dtype_size # dtype of tokens after the header

        os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
        print(f"writing {len(toks):,} tokens to {filename}")
        with open(filename, "wb") as f:
            f.write(header.tobytes())
            f.write(toks.tobytes())

    @staticmethod
    def read_datafile_tokens_memmap(path: Union[str, Path], dtype: Optional[np.dtype] = None) -> np.memmap:
        """Memory-map the token payload after the 256*4 byte header as uint16."""
        path = os.fspath(path)
        header_bytes = 256 * 4
        nbytes = PrecachedDatasetMixin.read_datafile_token_dtype(path)
        if dtype is None:
            if nbytes == 1:
                dtype = np.uint8
            elif nbytes == 2:
                dtype = np.uint16
            elif nbytes == 4:
                dtype = np.uint32
            elif nbytes == 8:
                dtype = np.uint64
            else:
                raise ValueError(f"Unsupported token dtype size in header: {nbytes}")
        else:
            assert dtype.itemsize == nbytes, \
                f"Intended dataset read size {dtype} does not match data storage type {nbytes} bytes"
        return np.memmap(path, mode="r", dtype=dtype, offset=header_bytes)

    @staticmethod
    def read_datafile_token_count(path: Union[str, Path]) -> int:
        """Read header[2] which stores token count."""
        with open(path, "rb") as f:
            header = np.frombuffer(f.read(256 * 4), dtype=np.int32)
        return int(header[2])

    @staticmethod
    def read_datafile_token_dtype(path: Union[str, Path]) -> int:
        """Read header[3] which stores token dtype"""
        with open(path, "rb") as f:
            header = np.frombuffer(f.read(256 * 4), dtype=np.int32)
        return int(header[3])

    @staticmethod
    def _tokenize_worker(
        doc: str,
        encode: Callable[[str], List[int]] = None,
        doc_separator: Optional[int] = None,
        dtype: np.dtype = np.uint16
    ) -> np.ndarray:
        toks = encode(doc)
        if doc_separator:
            toks.append(doc_separator)
        return np.asarray(toks, dtype=dtype)