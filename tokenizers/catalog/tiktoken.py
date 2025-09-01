import tiktoken
from typing import Union
from enum import Enum


class TikTokenModel(Enum):
    GPT2 = "gpt2"
    GPT4 = "cl100k_base"
    GPT5 = "o200k_base"
    GPTOSS = "o200k_harmony"
    cl100k_base = "cl100k_base"
    o200k_base = "o200k_base"
    o200k_harmony = "o200k_harmony"


def get_tiktoken_tokenizer(model: Union[str, TikTokenModel] = TikTokenModel.GPT2):
    """
    Get a tiktoken tokenizer for the specified model.
    
    Args:
        model: Either a string model name or TikTokenModel enum value
        
    Returns:
        tiktoken.Encoding: The tokenizer for the specified model
    """
    if isinstance(model, TikTokenModel):
        model_name = model.value
    else:
        model_name = model
    
    return tiktoken.get_encoding(model_name)