"""Rank-list fusion functions for hybrid retrieval."""

from __future__ import annotations

from collections.abc import Sequence


def reciprocal_rank_fusion(
    dense_documents: Sequence[dict[str, object]],
    bm25_documents: Sequence[dict[str, object]],
    k: int = 60,
    dense_weight: float = 1.0,
    bm25_weight: float = 1.0,
) -> list[dict[str, object]]:
    """Merge dense and BM25 rankings using Reciprocal Rank Fusion.

    Args:
        dense_documents: Dense candidates in descending relevance order.
        bm25_documents: BM25 candidates in descending relevance order.
        k: RRF rank constant. Larger values reduce the impact of high ranks.
        dense_weight: Weight applied to dense ranking contributions.
        bm25_weight: Weight applied to BM25 ranking contributions.

    Returns:
        De-duplicated candidates sorted by descending RRF score, then chunk ID
        for deterministic ordering.
    """
    if k < 0:
        raise ValueError("k must be non-negative")
    merged: dict[str, dict[str, object]] = {}
    scores: dict[str, float] = {}
    for documents, weight in ((dense_documents, dense_weight), (bm25_documents, bm25_weight)):
        for rank, document in enumerate(documents, start=1):
            chunk_id = str(document["chunk_id"])
            merged.setdefault(chunk_id, dict(document))
            scores[chunk_id] = scores.get(chunk_id, 0.0) + weight / (k + rank)
    return [
        {**merged[chunk_id], "rrf_score": scores[chunk_id]}
        for chunk_id in sorted(scores, key=lambda value: (-scores[value], value))
    ]
