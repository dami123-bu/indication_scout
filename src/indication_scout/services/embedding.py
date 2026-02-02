"""Embedding service for semantic search."""

import numpy as np


class EmbeddingService:
    """Service for generating and comparing embeddings."""

    async def embed_text(self, text: str) -> list[float]:
        """Generate embedding for text."""
        raise NotImplementedError

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        raise NotImplementedError

    def cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two embeddings."""
        a_arr = np.array(a)
        b_arr = np.array(b)
        return float(np.dot(a_arr, b_arr) / (np.linalg.norm(a_arr) * np.linalg.norm(b_arr)))
