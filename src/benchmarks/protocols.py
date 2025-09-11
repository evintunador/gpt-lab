from dataclasses import dataclass
from typing import List

@dataclass
class LogitMultipleChoiceItem:
    """
    Standardized data format for a single item in a logit-based multiple choice benchmark.
    """
    context: str
    choices: List[str]
    label: int


# In the future, we could add more protocols here
