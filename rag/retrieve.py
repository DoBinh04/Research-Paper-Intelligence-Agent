"""Hybrid retrieval orchestration while preserving the existing public API."""

from __future__ import annotations

from collections.abc import Mapping

from config.logging import get_logger
from config.settings import settings
from rag.bm25_retriever import get_bm25_retriever
from rag.dense_retriever import retrieve_dense
from rag.fusion import reciprocal_rank_fusion
from rag.query_rewriter import rewrite_query
from rag.reranker import rerank

logger = get_logger(__name__)


def retrieve_chunks(query: str, top_k: int = 5, filter_meta: dict | None = None, rewrite: bool = True) -> list[dict]:
    """Retrieve and rerank relevant paper chunks using hybrid search.

    Args:
        query: User's natural language search query.
        top_k: Maximum number of chunks to return after reranking.
        filter_meta: Optional ``paper_id``, ``section_type``, or
            ``section_heading`` filters.
        rewrite: Whether to rewrite the query before retrieval.

    Returns:
        The highest-ranked chunk dictionaries, including existing metadata and
        retrieval scores.
    """
    if top_k <= 0:
        return []
    search_query = rewrite_query(query) if rewrite else query
    filters: Mapping[str, object] | None = filter_meta
    dense = retrieve_dense(search_query, settings.dense_candidates, filters)
    logger.info("Dense retrieved: %d", len(dense))
    if settings.enable_hybrid:
        bm25 = get_bm25_retriever().search(search_query, settings.bm25_candidates, filters)
        logger.info("BM25 retrieved: %d", len(bm25))
        candidates = reciprocal_rank_fusion(
            dense, bm25, k=settings.rrf_k,
            dense_weight=settings.rrf_dense_weight, bm25_weight=settings.rrf_bm25_weight,
        )
    else:
        candidates = dense
    logger.info("After RRF: %d", len(candidates))
    ranked = rerank(search_query, candidates, top_k=top_k)
    logger.info("After rerank: %d", len(ranked))
    return ranked
