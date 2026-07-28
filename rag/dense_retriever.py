"""ChromaDB-backed dense candidate retrieval."""

from __future__ import annotations

from collections.abc import Mapping

from rag.embedder import embed_query
from rag.store import get_chunks_collection

SUPPORTED_FILTERS = frozenset({
    "paper_id",
    "section_type",
    "section_heading",
    "subsection_heading",
    "page",
})


def retrieve_dense(
    query: str,
    top_k: int,
    filter_meta: Mapping[str, object] | None = None,
) -> list[dict[str, object]]:
    """Retrieve dense candidates from ChromaDB without reranking.

    Args:
        query: Query text, already rewritten when query rewriting is enabled.
        top_k: Number of dense candidates to request.
        filter_meta: Optional exact-match filters for ``paper_id``,
            ``section_type``, and ``section_heading``.

    Returns:
        Candidate chunks with content, stable chunk IDs, metadata, and their
        Chroma cosine distances.
    """
    where = _to_chroma_filter(filter_meta)
    results = get_chunks_collection().query(
        query_embeddings=[embed_query(query)],
        n_results=top_k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    ids = results.get("ids", [[]])[0]
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]
    candidates: list[dict[str, object]] = []
    for index, document_id in enumerate(ids):
        metadata = metadatas[index] if index < len(metadatas) else {}
        candidates.append(_candidate(
            content=documents[index],
            metadata=metadata,
            chunk_id=metadata.get("chunk_id", document_id),
            distance=distances[index] if index < len(distances) else 1.0,
        ))
    return candidates


def _to_chroma_filter(filter_meta: Mapping[str, object] | None) -> dict[str, object] | None:
    clauses = [
        {key: value}
        for key, value in (filter_meta or {}).items()
        if key in SUPPORTED_FILTERS and value is not None
    ]
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses} if clauses else None


def _candidate(
    content: object,
    metadata: Mapping[str, object],
    chunk_id: object,
    distance: object,
) -> dict[str, object]:
    return {
        "content": str(content), "paper_id": str(metadata.get("paper_id", "")),
        "paper_title": str(metadata.get("paper_title", "")),
        "section_type": str(metadata.get("section_type", "")),
        "section_heading": str(metadata.get("section_heading", "")),
        "subsection_heading": str(metadata.get("subsection_heading", "")),
        "heading_path": str(metadata.get("heading_path", "")),
        "chunk_index": int(metadata.get("chunk_index", 0)),
        "page": int(metadata.get("page", 0)),
        "chunk_id": str(chunk_id),
        "distance": float(distance),
    }
