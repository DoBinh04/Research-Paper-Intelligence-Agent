"""In-memory BM25 index synchronized with persisted Chroma chunks."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from functools import lru_cache

from rank_bm25 import BM25Okapi

from rag.store import get_chunks_collection

_FILTER_KEYS = frozenset({
    "paper_id",
    "section_type",
    "section_heading",
    "subsection_heading",
    "page",
})
_TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)


class BM25Retriever:
    """Maintain a lazily loaded, in-memory BM25 index over paper chunks."""

    def __init__(self) -> None:
        """Initialize an empty index that loads persisted chunks on first use."""
        self._documents: dict[str, dict[str, object]] = {} #dict contain all chunks
        self._index: BM25Okapi | None = None #statistic that BM25 needs to calculate when retrieve
        self._index_ids: list[str] = []  #list contain ids
        self._loaded = False

    def load_existing_chunks(self) -> None:
        """Load all currently persisted Chroma chunks into the BM25 corpus."""
        if self._loaded:
            return
        collection = get_chunks_collection()
        total = collection.count()
        if total:
            stored = collection.get(limit=total, include=["documents", "metadatas"])
            ids = stored.get("ids", [])
            documents = stored.get("documents", [])
            metadatas = stored.get("metadatas", [])
            for index, stored_id in enumerate(ids):
                metadata = metadatas[index] if index < len(metadatas) else {}
                self._documents[str(metadata.get("chunk_id", stored_id))] = _make_candidate(
                    documents[index], metadata, str(metadata.get("chunk_id", stored_id))
                )
        self._loaded = True
        self.rebuild()

    def add_chunks(self, chunks: Iterable[Mapping[str, object]]) -> None:
        """Add chunks to the corpus and rebuild the BM25 index once.

        Args:
            chunks: Chunk dictionaries in the same public format as dense
                retrieval candidates.
        """
        self.load_existing_chunks()
        for chunk in chunks:
            chunk_id = str(chunk["chunk_id"])
            self._documents[chunk_id] = dict(chunk)
        self.rebuild()

    def remove_paper(self, paper_id: str) -> None:
        """Remove every indexed chunk belonging to a paper and rebuild.

        Args:
            paper_id: Identifier of the paper whose chunks should be removed.
        """
        self.load_existing_chunks()

        before = len(self._documents)

        self._documents = {
            chunk_id: chunk for chunk_id, chunk in self._documents.items()
            if chunk.get("paper_id") != paper_id
        }

        after = len(self._documents)
        if after != before:
            self.rebuild()

    def replace_paper(self, paper_id: str, chunks: Iterable[Mapping[str, object]]) -> None:
        """Atomically replace one paper's chunks and rebuild the index once.

        Args:
            paper_id: Identifier of the paper being re-ingested.
            chunks: New chunks to index for that paper.
        """
        self.load_existing_chunks()
        self._documents = {
            chunk_id: chunk for chunk_id, chunk in self._documents.items()
            if chunk.get("paper_id") != paper_id
        }
        for chunk in chunks:
            self._documents[str(chunk["chunk_id"])] = dict(chunk)
        self.rebuild()

    def rebuild(self) -> None:
        """Rebuild the rank-bm25 model from the current in-memory corpus."""
        self._index_ids = sorted(self._documents)
        tokenized = [_tokenize(str(self._documents[chunk_id]["content"])) for chunk_id in self._index_ids]
        self._index = BM25Okapi(tokenized) if tokenized else None

    def search(
        self,
        query: str,
        top_k: int,
        filter_meta: Mapping[str, object] | None = None,
    ) -> list[dict[str, object]]:
        """Rank matching chunks with BM25, applying metadata filters first.

        Args:
            query: Query text, already rewritten when query rewriting is enabled.
            top_k: Maximum number of lexical candidates to return.
            filter_meta: Optional exact-match metadata filters shared with dense
                retrieval.

        Returns:
            BM25-ranked chunk candidates with a ``bm25_score`` field.
        """
        self.load_existing_chunks()
        if not self._index or top_k <= 0:
            return []
        filters = {key: value for key, value in (filter_meta or {}).items() if key in _FILTER_KEYS and value is not None}
        scores = self._index.get_scores(_tokenize(query))
        scored = [
            (float(scores[index]), chunk_id)
            for index, chunk_id in enumerate(self._index_ids)
            if all(self._documents[chunk_id].get(key) == value for key, value in filters.items())
        ]
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [{**self._documents[chunk_id], "bm25_score": score} for score, chunk_id in scored[:top_k]]


def _tokenize(text: str) -> list[str]:
    return _TOKEN_PATTERN.findall(text.lower())


def _make_candidate(content: object, metadata: Mapping[str, object], chunk_id: str) -> dict[str, object]:
    return {
        "content": str(content), "paper_id": str(metadata.get("paper_id", "")),
        "paper_title": str(metadata.get("paper_title", "")),
        "section_type": str(metadata.get("section_type", "")),
        "section_heading": str(metadata.get("section_heading", "")),
        "subsection_heading": str(metadata.get("subsection_heading", "")),
        "heading_path": str(metadata.get("heading_path", "")),
        "chunk_index": int(metadata.get("chunk_index", 0)), "page": int(metadata.get("page", 0)),
        "chunk_id": chunk_id,
    }


@lru_cache
def get_bm25_retriever() -> BM25Retriever:
    """Return the process-wide BM25 retriever singleton.

    Returns:
        The shared BM25 retriever used by ingestion and retrieval.
    """
    return BM25Retriever()