"""LLM client wrapper."""

from typing import Any


class LLMClient:
    """Wrapper for LLM API calls."""

    def __init__(self, model: str = "gpt-4"):
        self.model = model

    async def complete(self, prompt: str, **kwargs: Any) -> str:
        """Generate a completion for the given prompt."""
        raise NotImplementedError

    async def embed(self, text: str) -> list[float]:
        """Generate embeddings for the given text."""
        raise NotImplementedError
