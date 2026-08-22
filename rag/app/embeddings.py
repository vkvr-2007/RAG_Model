from __future__ import annotations

import hashlib
import logging

import numpy as np


class HashEmbedder:
    """Dependency-free deterministic fallback for local-only indexing and retrieval."""
    dimension = 384

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in text.lower().split():
                index = int(hashlib.sha256(token.encode()).hexdigest(), 16) % self.dimension
                vectors[row, index] += 1
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.maximum(norms, 1e-12)


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        getter = getattr(self.model, "get_embedding_dimension", None) or self.model.get_sentence_embedding_dimension
        self.dimension = getter()

    def encode(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False).astype(np.float32)


def build_embedder(backend: str, model_name: str):
    if backend == "hash":
        return HashEmbedder()
    if backend != "sentence_transformers":
        raise ValueError("RAG_EMBEDDING_BACKEND must be 'sentence_transformers' or 'hash'")
    try:
        return SentenceTransformerEmbedder(model_name)
    except Exception as error:  # pragma: no cover - depends on local model availability
        logger = logging.getLogger(__name__)
        logger.warning("Falling back to hash embeddings because '%s' could not be loaded: %s", model_name, error)
        return HashEmbedder()
