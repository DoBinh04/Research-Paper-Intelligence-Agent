"""BGE / sentence-transformers embedding wrapper."""

from typing import List

import numpy as np

from config.logging import get_logger
from config.settings import settings

logger = get_logger(__name__)

_model = None
_model_type: str | None = None


def _load_model():
    """
    Load and cache the embedding model.

    The function lazily initializes the embedding model on its first call and
    reuses the cached instance for subsequent calls.

    Loading priority:
        1. BGE-M3 (`BAAI/bge-m3`) if `settings.use_bge_m3` is enabled.
        2. Fallback to a SentenceTransformer model specified by
           `settings.embedding_model` if BGE-M3 is unavailable or fails to load.

    Returns:
        tuple:
            - object: Loaded embedding model instance.
            - str: Model type identifier. One of:
                - "bge-m3"
                - "sentence-transformers"
    """
    global _model, _model_type
    if _model is not None:
        return _model, _model_type

    if settings.use_bge_m3:
        try:
            from FlagEmbedding import BGEM3FlagModel

            _model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=False)
            _model_type = "bge-m3"
            logger.info("Loaded BGE-M3 embedding model")
            return _model, _model_type
        except Exception as exc:
            logger.warning("BGE-M3 unavailable (%s), falling back to sentence-transformers", exc)

    from sentence_transformers import SentenceTransformer

    _model = SentenceTransformer(settings.embedding_model)
    _model_type = "sentence-transformers"
    logger.info("Loaded embedding model: %s", settings.embedding_model)
    return _model, _model_type


def embed_texts(texts: List[str], batch_size: int = 32) -> List[List[float]]:
    """
    Generate dense embedding vectors for multiple text inputs.

    The function automatically selects the embedding backend based on the
    loaded model:
        - BGE-M3 (`BGEM3FlagModel`) if available.
        - SentenceTransformer otherwise.

    Args:
        texts: A list of input texts to embed.
        batch_size: Number of texts processed in each inference batch.
            Defaults to 32.

    Returns:
        List[List[float]]:
            A list of dense embedding vectors, where each vector corresponds
            to an input text in the same order.

    Notes:
        - Returns an empty list if `texts` is empty.
        - For BGE-M3, only the dense embeddings (`dense_vecs`) are returned.
        - NumPy arrays are converted to standard Python lists for
          compatibility with vector databases and JSON serialization.
    """
    if not texts:
        return []

    model, model_type = _load_model()
    if model_type == "bge-m3":
        output = model.encode(texts, batch_size=batch_size, max_length=8192)
        vectors = output["dense_vecs"]
        return [v.tolist() if hasattr(v, "tolist") else list(v) for v in vectors]

    vectors = model.encode(texts, batch_size=batch_size, show_progress_bar=False)
    if isinstance(vectors, np.ndarray):
        return vectors.tolist()
    return [v.tolist() if hasattr(v, "tolist") else list(v) for v in vectors]


def embed_query(query: str) -> List[float]:
    """
    Generate a dense embedding vector for a single query.

    This is a convenience wrapper around `embed_texts()` for embedding
    individual search queries.

    Args:
        query: The query text to embed.

    Returns:
        List[float]:
            The dense embedding vector representing the input query.
    """
    return embed_texts([query])[0]
