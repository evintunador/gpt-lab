from abc import abstractmethod, ABC


class LLMClient(ABC):
    """
    Minimal LLM protocol. Plug in any provider that returns a string completion.
    """
    refine_user_prompt = \
"""{user_prompt}

Fix the prior code to address the following errors. Output ONLY a single complete Python file (no backticks, no commentary).

Errors to fix:
{error_summary}

Prior code:
{prior_code}
"""
    
    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str) -> str: 
        raise NotImplementedError("Provide a real LLM client implementing LLMClient.generate().")
    
    @abstractmethod
    def refine(self, system_prompt: str, user_prompt: str, prior_code: str, error_summary: str) -> str: 
        raise NotImplementedError("Provide a real LLM client implementing LLMClient.refine().")

    def strip_code_fences(self, text: str) -> str:
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


