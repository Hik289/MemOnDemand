"""General embeddings client with batching, retries, and L2 normalization."""

from __future__ import annotations

import logging
import os
from typing import Iterable, List

import numpy as np

from memondemand.core.api_adapter import embed


logger = logging.getLogger(__name__)


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def embed_batch(
    texts: List[str],
    batch_size: int = 128,
    *,
    timeout: float = 90.0,
    max_retries: int = 5,
) -> np.ndarray:
    """Embed a list of texts through the configured general endpoint."""
    if not texts:
        dim = int(os.environ.get("MEMONDEMAND_EMBED_DIM", "0"))
        return np.zeros((0, dim), dtype=np.float32)
    rows = []
    for start in range(0, len(texts), batch_size):
        stop = min(start + batch_size, len(texts))
        result = embed(
            texts[start:stop],
            timeout=timeout,
            max_retries=max_retries,
        )
        rows.extend(result["vectors"])
        logger.info("general embedding progress: %d/%d", stop, len(texts))
    vectors = np.asarray(rows, dtype=np.float32)
    if vectors.ndim != 2:
        raise RuntimeError("General embedding endpoint returned a non-matrix result")
    return _normalize(vectors).astype(np.float32)


class GeneralEmbedder:
    """Sentence-transformers-compatible wrapper for the general endpoint."""

    def __init__(self, model_name: str | None = None, dim: int | None = None):
        self.model_name = model_name or os.environ.get(
            "MEMONDEMAND_EMBED_API_MODEL", ""
        )
        self.dim = dim or int(os.environ.get("MEMONDEMAND_EMBED_DIM", "1536"))

    def encode(self, texts: Iterable[str], **kwargs) -> np.ndarray:
        values = list(texts)
        batch_size = int(kwargs.pop("batch_size", 128))
        vectors = embed_batch(values, batch_size=batch_size)
        if len(vectors):
            self.dim = int(vectors.shape[1])
        return vectors


if __name__ == "__main__":
    encoder = GeneralEmbedder()
    matrix = encoder.encode(["general embedding smoke test"])
    print(f"shape={matrix.shape} model={encoder.model_name}")
