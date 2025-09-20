import json
import os
from collections import defaultdict
from dataclasses import asdict
from typing import List, Optional

from datasets import load_dataset
from torch.utils.data import Dataset
from tqdm import tqdm

from data_sources.catalog_utils import Split
from benchmarks.catalog.multiple_choice import MultipleChoiceItem


class WikiQADataset(Dataset):
    """
    The WikiQA dataset from Microsoft, with support for memory-efficient streaming.

    This dataset is structured with one question-candidate answer pair per row.
    This class pre-processes the data by grouping all candidate answers for a
    given question into a single `MultipleChoiceItem`. It then saves the processed
    data to a local cache and can either load it all into RAM or read from the
    cache file on-demand.

    Reference: https://huggingface.co/datasets/microsoft/wiki_qa
    """

    def __init__(
        self,
        split: Split = Split.VAL,
        cache_dir: Optional[str] = None,
        in_memory: bool = False,
        limit: Optional[int] = None,
    ):
        """
        Initializes the WikiQADataset.

        Args:
            split: Which data split to use (train/val/test).
            cache_dir: Directory to cache the downloaded and processed data.
            in_memory: If True, loads the entire processed dataset into RAM.
                       If False, reads from the cache file for each item lookup.
            limit: Maximum number of questions to load.
        """
        self.split = split
        if cache_dir is None:
            cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "gpt-lab", "wiki_qa")
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_dir = cache_dir

        self.processed_data_path = os.path.join(self.cache_dir, f"{split.value}.jsonl")
        self.processed_index_path = os.path.join(self.cache_dir, f"{split.value}.index")

        if not os.path.exists(self.processed_data_path):
            self._process_and_cache_raw_data()

        self.in_memory = in_memory
        if self.in_memory:
            self.data: List[MultipleChoiceItem] = []
            with open(self.processed_data_path, "r") as f:
                for line in f:
                    self.data.append(MultipleChoiceItem(**json.loads(line)))
        else:
            self.line_offsets = []
            with open(self.processed_index_path, "r") as f:
                for line in f:
                    self.line_offsets.append(int(line))
            self.data_file_handle = open(self.processed_data_path, "r")
        
        # Apply limit if provided
        if limit is not None:
            if self.in_memory:
                self.data = self.data[:limit]
            else:
                self.line_offsets = self.line_offsets[:limit]

    def _process_and_cache_raw_data(self):
        """Loads raw data, processes it, and saves to a local cache."""
        print(f"Processing and caching WikiQA '{self.split.value}' split...")
        raw_dataset = load_dataset("microsoft/wiki_qa", split=self.split.value)

        questions = defaultdict(lambda: {"question": "", "answers": [], "labels": []})
        for example in tqdm(raw_dataset, desc="Grouping questions"):
            qid = example["question_id"]
            questions[qid]["question"] = example["question"]
            questions[qid]["answers"].append(example["answer"])
            questions[qid]["labels"].append(example["label"])

        with open(self.processed_data_path, "w") as f_data, \
             open(self.processed_index_path, "w") as f_index:
            for content in tqdm(questions.values(), desc="Writing to cache"):
                try:
                    correct_index = content["labels"].index(1)
                    item = MultipleChoiceItem(
                        context=content["question"],
                        choices=content["answers"],
                        label=correct_index,
                    )
                    item_dict = asdict(item)
                    json_str = json.dumps(item_dict) + "\n"

                    offset = f_data.tell()
                    f_data.write(json_str)
                    f_index.write(f"{offset}\n")
                except ValueError:
                    pass  # Skip questions with no correct answer

    def __len__(self) -> int:
        return len(self.data) if self.in_memory else len(self.line_offsets)

    def __getitem__(self, idx: int) -> MultipleChoiceItem:
        if self.in_memory:
            return self.data[idx]
        
        offset = self.line_offsets[idx]
        self.data_file_handle.seek(offset)
        line = self.data_file_handle.readline()
        return MultipleChoiceItem(**json.loads(line))

    def __del__(self):
        if hasattr(self, "data_file_handle"):
            self.data_file_handle.close()


"""
Example of how this class might be used in a script:

from src.benchmarks.catalog.multiple_choice import MultipleChoiceBenchmark
# from some_model_file import MyModel

# 1. Instantiate the model
# model = MyModel()

# 2. Instantiate the dataset (memory-efficient) and benchmark runner
dataset = WikiQADataset(split="test", in_memory=False)
benchmark = MultipleChoiceBenchmark(model)

# 3. Run the evaluation
results = benchmark.run(dataset, limit=100) # Limit for a quick test
print("WikiQA Results:", results)
"""
