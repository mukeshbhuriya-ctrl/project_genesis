"""
Vector serialization, cosine similarity, and embedding management.
"""

from __future__ import annotations

import math
import struct
from typing import Sequence

from genesis.core import get_settings
from genesis.core.logging import get_logger
from genesis.models.ollama_backend import OllamaProvider

log = get_logger("memory.embeddings")


def pack_vector(vector: Sequence[float]) -> bytes:
    """Pack a floating point vector into compact IEEE 754 float32 bytes."""
    return struct.pack(f"{len(vector)}f", *vector)


def unpack_vector(blob: bytes) -> list[float]:
    """Unpack binary float32 bytes into a list of floats."""
    count = len(blob) // 4
    return list(struct.unpack(f"{count}f", blob))


def cosine_similarity(v1: Sequence[float], v2: Sequence[float]) -> float:
    """Compute cosine similarity between two vectors (-1.0 to 1.0)."""
    if len(v1) != len(v2) or not v1:
        return 0.0

    dot = 0.0
    norm1 = 0.0
    norm2 = 0.0

    for a, b in zip(v1, v2):
        dot += a * b
        norm1 += a * a
        norm2 += b * b

    if norm1 <= 0.0 or norm2 <= 0.0:
        return 0.0

    return dot / (math.sqrt(norm1) * math.sqrt(norm2))


class EmbeddingEngine:
    """High-level client for generating and storing vector embeddings."""

    def __init__(self, provider: OllamaProvider | None = None, model: str | None = None):
        self.provider = provider or OllamaProvider()
        settings = get_settings()
        self.model = model or settings.embedding_model or "nomic-embed-text"

    async def get_embedding(self, text: str) -> list[float]:
        """Generate embedding vector for a single string."""
        if not text.strip():
            return []
        vectors = await self.provider.embed(texts=[text], model=self.model)
        return vectors[0] if vectors else []

    async def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple strings in batch."""
        clean_texts = [t for t in texts if t.strip()]
        if not clean_texts:
            return []
        return await self.provider.embed(texts=clean_texts, model=self.model)
