from typing import Optional

from .abstract_base import LLMClient
from .openai import OpenAILLM
from .anthropic import AnthropicLLM


def create_llm(model: str, api_key: Optional[str] = None) -> LLMClient:
    """
    Factory that picks a client from a single model string.
    Supports:
        - 'openai/<model>' or bare '<model>' (if known) -> OpenAI
        - 'antrhopic/<model>' or bare '<model>' (if known) -> Anthropic
    If provider prefix is omitted, heuristics pick a provider by model name.
    """
    m = (model or "").strip()
    if not m:
        raise ValueError("Model string must be non-empty")
    
    if "/" in m:
        provider, short = m.split("/", 1)
    else:
        short_lower = m.lower()
        if short_lower.startswith(("claude", "haiku", "sonnet", "opus")):
            provider, short = "anthropic", m
        else:
            # default to OpenAI for now
            provider, short = "openai", m
    
    normalized = f"{provider}/{short}"
    if provider == "openai":
        return OpenAILLM(api_key=api_key, model=normalized)
    elif provider == "anthropic":
        return AnthropicLLM(api_key=api_key, model=normalized)
    raise ValueError(f"Unsupported provider in model string: {model!r}")