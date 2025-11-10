import tempfile
import numpy as np
import pytest
from pathlib import Path
from src.gpt_lab.data_sources.catalog_utils import BinaryShardIO, SequentialPretokenizedDatasetMixin, Split
import torch


def _test_tokenizer_encode_fn(s: str):
    return [ord(c) for c in s]


def test_pick_token_dtype():
    assert BinaryShardIO.pick_token_dtype(255) == np.uint8
    assert BinaryShardIO.pick_token_dtype(256) == np.uint16
    assert BinaryShardIO.pick_token_dtype(65535) == np.uint16
    assert BinaryShardIO.pick_token_dtype(65536) == np.uint32
    assert BinaryShardIO.pick_token_dtype(2**32 - 1) == np.uint32
    assert BinaryShardIO.pick_token_dtype(2**32) == np.uint64


@pytest.mark.parametrize(
    "dtype",
    [np.uint8, np.uint16, np.uint32, np.uint64],
)
def test_datafile_write_read_roundtrip(dtype):
    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = Path(temp_dir) / "test_shard.bin"
        num_tokens = 1000
        if dtype == np.uint8:
            max_val = 255
        elif dtype == np.uint16:
            max_val = 65535
        elif dtype == np.uint32:
            max_val = 2**32 - 1
        else:
            max_val = 2**64 - 1

        original_tokens = np.random.randint(0, min(max_val, 50000), size=num_tokens, dtype=dtype)

        BinaryShardIO.write_datafile(file_path, original_tokens)

        # Test read_datafile_token_count
        token_count = BinaryShardIO.read_datafile_token_count(file_path)
        assert token_count == num_tokens

        # Test read_datafile_token_dtype
        dtype_size = BinaryShardIO.read_datafile_token_dtype(file_path)
        assert dtype_size == np.dtype(dtype).itemsize

        # Test read_datafile_tokens_memmap
        memmapped_tokens = BinaryShardIO.read_datafile_tokens_memmap(file_path, dtype=dtype)
        assert np.array_equal(original_tokens, memmapped_tokens)


class TestSequentialPretokenizedDatasetMixin:
    class Dataset(SequentialPretokenizedDatasetMixin):
        def __init__(self, save_dir, **kwargs):
            super().__init__(save_dir=save_dir, **kwargs)

    @pytest.fixture
    def dataset_path(self, tmp_path):
        return tmp_path / "dataset"

    def test_init(self, dataset_path):
        assert not dataset_path.exists()
        self.Dataset(save_dir=dataset_path)
        assert dataset_path.exists()
        assert dataset_path.is_dir()

    def test_build_and_has_cache(self, dataset_path):
        dataset = self.Dataset(save_dir=dataset_path, shard_size=100)

        assert not dataset.has_cache()

        doc_iter = ["hello world"] * 10
        dataset.build_cache(doc_iter, _test_tokenizer_encode_fn, doc_separator=ord("\n"), token_dtype=np.uint8)

        assert dataset.has_cache()

        val_files = sorted(dataset_path.glob("*_val_*.bin"))
        train_files = sorted(dataset_path.glob("*_train_*.bin"))

        assert len(val_files) == 1
        assert len(train_files) == 1

    def test_setup_cache_index_len_getitem(self, dataset_path):
        shard_size = 101
        seq_len = 10
        dataset = self.Dataset(save_dir=dataset_path, shard_size=shard_size)

        docs = ["a" * 50, "b" * 50, "c" * 50, "d" * 50, "e" * 50]
        total_tokens = sum(len(d) for d in docs)

        dataset.build_cache(docs, _test_tokenizer_encode_fn, token_dtype=np.uint8)

        # Test val split
        dataset.setup_cache_index(Split.VAL, seq_len)
        assert len(dataset) == 101 // seq_len

        val_data = np.concatenate([np.full(50, ord("a")), np.full(50, ord("b")), np.full(1, ord("c"))])
        for i in range(len(dataset)):
            start = i * seq_len
            end = start + seq_len
            expected = torch.from_numpy(val_data[start:end])
            assert torch.equal(dataset[i], expected.to(torch.long))

        # Test train split
        dataset.setup_cache_index(Split.TRAIN, seq_len)
        num_train_tokens = total_tokens - 101
        assert len(dataset) == num_train_tokens // seq_len

        train_data = np.concatenate(
            [np.full(49, ord("c")), np.full(50, ord("d")), np.full(2, ord("e")), np.full(48, ord("e"))]
        )

        for i in range(len(dataset)):
            start = i * seq_len
            end = start + seq_len
            expected = torch.from_numpy(train_data[start:end])
            assert torch.equal(dataset[i], expected.to(torch.long))

        # Test slicing across shard boundaries in train split
        i = 10
        start = i * seq_len
        end = start + seq_len
        expected = torch.from_numpy(train_data[start:end])
        assert torch.equal(dataset[i], expected.to(torch.long))
        assert (dataset[i] == ord("e")).all()

    def test_raises_on_uninitialized(self, dataset_path):
        dataset = self.Dataset(save_dir=dataset_path)
        with pytest.raises(RuntimeError, match="Cache index not initialized"):
            len(dataset)
        with pytest.raises(RuntimeError, match="Cache index not initialized"):
            dataset[0]

    def test_getitem_out_of_bounds(self, dataset_path):
        dataset = self.Dataset(save_dir=dataset_path, shard_size=50)
        docs = ["a" * 100]
        dataset.build_cache(docs, _test_tokenizer_encode_fn, token_dtype=np.uint8)

        dataset.setup_cache_index(Split.VAL, seq_len=10)

        with pytest.raises(IndexError):
            dataset[len(dataset)]
        with pytest.raises(IndexError):
            dataset[-1]
