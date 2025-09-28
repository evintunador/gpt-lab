import os
import re
import xml.etree.ElementTree as ET
from typing import List, Literal, Optional

from torch.utils.data import Dataset

from gpt_lab.data_sources.catalog_utils import download_file
from gpt_lab.benchmarks.catalog.fill_in_the_blank import FillInTheBlankItem


# The different answer formats we can extract from the dataset
Target = Literal["number", "formula", "full_answer", "answer_formula"]


class ASDivDataset(Dataset):
    """
    The Academia Sinica Diverse MWP Dataset (ASDiv).

    This class handles downloading and parsing the dataset. It can be configured
    to create prompts and answers for different evaluation targets.

    Reference: https://github.com/chaochun/nlu-asdiv-dataset
    """
    
    URL = "https://raw.githubusercontent.com/chaochun/nlu-asdiv-dataset/master/dataset/ASDiv.xml"

    def __init__(
        self,
        target: Target = "number",
        grade_levels: Optional[List[int]] = None,
        cache_dir: Optional[str] = None,
        limit: Optional[int] = None,
    ):
        """
        Initializes the ASDivDataset.

        Args:
            target: Which part of the data to use as the answer.
                - "number": The numerical answer (e.g., "9").
                - "formula": The mathematical formula (e.g., "7+2=9").
                - "full_answer": The full text of the answer field (e.g., "9 (apples)").
                - "answer_formula": Both answer and formula, concatenated.
            grade_levels: An optional list of grade levels to filter the dataset by.
            cache_dir: Directory to cache the downloaded XML file.
            limit: Maximum number of examples to load.
        """
        self.target = target

        if cache_dir is None:
            cache_dir = os.path.join("data", ".cache", "asdiv")
        os.makedirs(cache_dir, exist_ok=True)
        
        self.cache_path = os.path.join(cache_dir, "ASDiv.xml")
        if not os.path.exists(self.cache_path):
            print(f"Downloading ASDiv dataset to {self.cache_path}...")
            download_file(self.URL, self.cache_path)

        self.problems = self._parse_xml(self.cache_path)
        
        if grade_levels:
            self.problems = [
                p for p in self.problems if p["grade"] in grade_levels
            ]

        if limit is not None:
            self.problems = self.problems[:limit]

    def _parse_xml(self, xml_path: str) -> List[dict]:
        """Parses the ASDiv XML file into a list of problem dictionaries."""
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        problems = []
        for problem_set in root.findall('ProblemSet'):
            for problem in problem_set.findall('Problem'):
                problems.append({
                    "id": problem.get('ID'),
                    "grade": int(problem.get("Grade")),
                    "body": problem.find('Body').text,
                    "question": problem.find('Question').text,
                    "solution_type": problem.find('Solution-Type').text,
                    "answer": problem.find('Answer').text,
                    "formula": problem.find('Formula').text,
                })
        return problems

    def _get_number_from_answer(self, answer_text: str) -> str:
        """Extracts the leading number from the answer string."""
        match = re.match(r'[\d\.]+', answer_text)
        return match.group(0) if match else ""

    def __len__(self) -> int:
        return len(self.problems)

    def __getitem__(self, idx: int) -> FillInTheBlankItem:
        problem = self.problems[idx]
        
        # The prompt is always the body plus the question
        prompt = f"{problem['body']} {problem['question']}"
        
        # The answer depends on the configured target
        answer = ""
        if self.target == "number":
            answer = self._get_number_from_answer(problem["answer"])
        elif self.target == "formula":
            answer = problem["formula"]
        elif self.target == "full_answer":
            answer = problem["answer"]
        elif self.target == "answer_formula":
            answer = f'{problem["answer"]} ({problem["formula"]})'
            
        return FillInTheBlankItem(prompt=prompt, answer=answer)


"""
Example of how this class might be used in a script:

from gpt_lab.benchmarks.catalog.fill_in_the_blank import FillInTheBlankBenchmark
from some_model_file import MyModel

model = MyModel()

dataset_num = ASDivDataset(target="number", grade_levels=[3, 4])
benchmark_num = FillInTheBlankBenchmark(model)
results_num = benchmark_num.run(dataset_num)
"""