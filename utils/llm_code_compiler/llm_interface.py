from typing import Protocol, Optional
import os

from litellm import completion

from llm_code_compiler.catalog import OpenAILLM, AnthropicLLM


class LLMClient(Protocol):
    """
    Minimal LLM protocol. Plug in any provider that returns a string completion.
    """
    def generate(self, system_prompt: str, user_prompt: str) -> str: ...
    def refine(self, system_prompt: str, user_prompt: str, prior_code: str, error_summary: str) -> str: ...


def _strip_code_fences(text: str) -> str:
    if text is None:
        return ""
    # Remove triple backtick fences if present
    if "```" in text:
        # take inside of first/last fence as best-effort
        parts = text.split("```")
        # odd indices are fenced blocks; prefer first fenced block
        for i in range(1, len(parts), 2):
            if parts[i].strip():
                return parts[i].lstrip("python").lstrip()
        # fallback to concat of all non-fence segments
        return "".join(p for i, p in enumerate(parts) if i % 2 == 0)
    return text


class DummyLLM:
    """
    Default stub to make wiring explicit. Replace with a real adapter (e.g., OpenAI, Anthropic, local).
    """
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        raise NotImplementedError("Provide a real LLM client implementing LLMClient.generate().")

    def refine(self, system_prompt: str, user_prompt: str, prior_code: str, error_summary: str) -> str:
        raise NotImplementedError("Provide a real LLM client implementing LLMClient.refine().")


refine_user_prompt = \
"""{user_prompt}

Fix the prior code to address the following errors. Output ONLY a single complete Python file (no backticks, no commentary).

Errors to fix:
{error_summary}

Prior code:
{prior_code}
"""


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