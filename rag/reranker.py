"""Cohere reranker with a fused-rank fallback."""

from __future__ import annotations

import cohere

from config.logging import get_logger
from config.settings import settings

logger = get_logger(__name__)


def rerank(query: str, documents: list[dict], top_k: int = 5) -> list[dict]:
    """Rerank fused candidates using Cohere, preserving fusion order on fallback.

    Args:
        query: Query text used by the cross-encoder.
        documents: Fused candidates containing at least ``content``.
        top_k: Maximum number of final documents to return.

    Returns:
        Top candidates augmented with a ``relevance_score``. When Cohere is
        unavailable, candidates are ordered by their RRF score.
    """
    if not documents:
        return []
    if not settings.cohere_api_key:
        logger.warning("No COHERE_API_KEY - using fused retrieval order")
        ordered = sorted(
            documents,
            key=lambda document: (
                -float(document.get("rrf_score", 0.0)),
                float(document.get("distance", 1.0)),
                str(document.get("chunk_id", "")),
            ),
        )
        return [
            {
                **document,
                "relevance_score": float(
                    document.get("rrf_score", 1.0 - float(document.get("distance", 0.5)))
                ),
            }
            for document in ordered[:top_k]
        ]

    client = cohere.ClientV2(api_key=settings.cohere_api_key)
    try:
        response = client.rerank(
            model="rerank-english-v3.0",
            query=query,
            documents=[str(document["content"]) for document in documents],
            top_n=min(top_k, len(documents)),
        )
        ranked: list[dict] = []
        for item in response.results:
            document = documents[item.index].copy()
            document["relevance_score"] = float(item.relevance_score)
            ranked.append(document)
        return ranked
    except Exception as exc:
        logger.error("Cohere rerank failed: %s", exc)
        return documents[:top_k]
