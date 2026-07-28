"""Heading-bounded semantic chunking for research papers."""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict

import numpy as np
import tiktoken

from config.logging import get_logger

logger = get_logger(__name__)

MAX_TOKENS = 320
MIN_TOKENS = 90
OVERLAP_TOKENS = 30
SEMANTIC_BREAK_THRESHOLD = 0.42
KEEP_INTACT = {"THEOREM", "PROOF", "TABLE"}
_enc = tiktoken.get_encoding("o200k_base")


def count_tokens(text: str) -> int:
    return len(_enc.encode(text))


def _units(text: str) -> list[str]:
    """
    Split raw text into token-bounded semantic units.

    Args:
        text: Raw document text.

    Returns:
        A list of text units, each containing no more than ``MAX_TOKENS``
        tokens while preserving sentence boundaries whenever possible.
    """
    paragraphs = [" ".join(part.split()) for part in re.split(r"\n\s*\n", text) if part.strip()]
    sentences = [sentence.strip() for paragraph in paragraphs for sentence in
                 re.split(r"(?<!Mr|Ms|Dr)(?<=[.!?])\s+(?=[A-Z0-9])", paragraph) if sentence.strip()]
    units: list[str] = []
    for sentence in sentences:
        if count_tokens(sentence) <= MAX_TOKENS:
            units.append(sentence)
            continue
        words = sentence.split()
        current: list[str] = []
        for word in words:
            if current and count_tokens(" ".join(current + [word])) > MAX_TOKENS:
                units.append(" ".join(current))
                current = []
            current.append(word)
        if current:
            units.append(" ".join(current))
    return units


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    """Caculate cosine similarity between two vectors"""
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denom) if denom else 1.0


def _semantic_vectors(units: list[str]) -> list[np.ndarray] | None:
    """Generate text embeddings from list of text units"""
    try:
        from rag.embedder import embed_texts
        return [np.asarray(vector, dtype=float) for vector in embed_texts(units)]
    except Exception as exc:
        logger.warning("semantic embeddings unavailable; using token-limit chunking: %s", exc)
        return None

def _tail_overlap(parts: list[str]) -> list[str]:
    """Extract the trailing text segments to use as overlap between chunks"""
    overlap: list[str] = []
    tokens = 0
    for part in reversed(parts):
        part_tokens = count_tokens(part)
        if overlap and tokens + part_tokens > OVERLAP_TOKENS:
            break
        overlap.insert(0, part)
        tokens += part_tokens
    return overlap


def _semantic_split(text: str, section_type: str, vectors: list[np.ndarray] | None = None) -> list[str]:
    """
    Split a section into semantically coherent chunks.

    The algorithm first divides the text into small semantic units
    (paragraphs or sentences), computes an embedding for each unit,
    and then merges consecutive units into chunks.

    A new chunk is created when either:
        1. Adding the next unit would exceed MAX_TOKENS.
        2. The current chunk already contains at least MIN_TOKENS and
           the semantic similarity between adjacent units falls below
           SEMANTIC_BREAK_THRESHOLD.

    To preserve context between neighbouring chunks, the tail of the
    previous chunk is copied into the beginning of the next chunk
    according to OVERLAP_TOKENS.

    Sections listed in KEEP_INTACT (e.g. theorem, proof, table) or
    sections whose length does not exceed MAX_TOKENS are returned
    unchanged.

    Args:
        text:
            Section content to split.

        section_type:
            Type of section (e.g. ABSTRACT, METHOD, THEOREM).

        vectors:
            Optional precomputed embedding vectors corresponding to
            semantic units. If omitted, embeddings are generated
            internally.

    Returns:
        A list of semantic chunks preserving topic continuity while
        satisfying the configured token limits.
    """
    if section_type in KEEP_INTACT or count_tokens(text) <= MAX_TOKENS:
        return [text]
    parts = _units(text)
    if not parts:
        return []
    vectors = vectors if vectors is not None else _semantic_vectors(parts)
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for index, part in enumerate(parts):
        part_tokens = count_tokens(part)
        similarity = _cosine(vectors[index - 1], vectors[index]) if vectors is not None and index else 1.0
        should_break = current and (current_tokens + part_tokens > MAX_TOKENS or (current_tokens >= MIN_TOKENS and similarity < SEMANTIC_BREAK_THRESHOLD))
        if should_break:
            chunks.append(" ".join(current))
            current = _tail_overlap(current)
            current_tokens = sum(count_tokens(item) for item in current)
        current.append(part)
        current_tokens += part_tokens
    if current:
        chunks.append(" ".join(current))
    return chunks


def _heading_metadata(path: list[str]) -> tuple[str, str, str]:
    section = path[0] if path else ""
    subsection = path[-1] if len(path) > 1 else ""
    return section, subsection, " > ".join(path)


def chunk_blocks(blocks: list[dict], paper_id: str, paper_title: str = "") -> list[dict]:
    """
    Convert parsed document blocks into heading-bounded semantic chunks.

    The function groups consecutive content blocks by their ``heading_path``,
    ensuring that text from different sections or sibling headings is never
    merged into the same chunk. Each heading group is concatenated and passed
    to ``_semantic_split()`` for semantic-aware chunking.

    Every generated chunk is enriched with metadata describing its source
    paper, section hierarchy, page number, chunk indices, and content type
    (e.g. mathematical expressions or tables), making it suitable for
    embedding and retrieval in a RAG pipeline.

    Args:
        blocks:
            List of parsed document blocks produced by the parser. Each block
            may contain fields such as ``content``, ``heading_path``,
            ``section_type``, ``page``, ``is_math``, and ``is_table``.
        paper_id:
            Unique identifier of the source paper. Used for metadata and
            deterministic chunk ID generation.

        paper_title:
            Title of the source paper stored as chunk metadata.
            Defaults to an empty string.

    Returns:
        A list of chunk dictionaries. Each dictionary contains the chunk
        content together with metadata including paper information, section
        hierarchy, page number, chunk indices, unique chunk ID, and flags
        indicating whether the chunk contains mathematical expressions or
        tables.
    """

    grouped: dict[tuple[str, ...], list[dict]] = defaultdict(list)  #Group content blocks by heading path.
    order: list[tuple[str, ...]] = [] #Store the chronological order of heading paths.
    for block in blocks:
        if block.get("is_heading"):
            continue
        content = (block.get("content") or "").strip()
        if not content:
            continue
        path = tuple(block.get("heading_path") or ([block.get("heading")] if block.get("heading") else []))
        if path not in grouped:
            order.append(path)
        grouped[path].append(block)

    chunks: list[dict] = []
    for path in order:
        blocks_in_heading = grouped[path]
        section_type = next((str(block.get("section_type") or "OTHER").upper() for block in blocks_in_heading if block.get("section_type")), "OTHER")
        page = blocks_in_heading[0].get("page_num", blocks_in_heading[0].get("page", 0))
        text = "\n\n".join(str(block["content"]).strip() for block in blocks_in_heading)
        parts = _semantic_split(text, section_type)
        section_heading, subsection_heading, heading_path = _heading_metadata(list(path))
        for heading_chunk_index, part in enumerate(parts):
            chunk_index = len(chunks)
            chunk_id = hashlib.sha256(f"{paper_id}:{heading_path}:{chunk_index}:{part}".encode()).hexdigest()[:16]
            chunks.append({
                "content": part, "paper_id": paper_id, "paper_title": paper_title,
                "section_type": section_type, "section_heading": section_heading,
                "subsection_heading": subsection_heading, "heading_path": heading_path,
                "chunk_index": chunk_index, "heading_chunk_index": heading_chunk_index,
                "chunk_id": chunk_id, "page": page,
                "is_math": any(bool(block.get("is_math")) for block in blocks_in_heading),
                "is_table": any(bool(block.get("is_table")) for block in blocks_in_heading),
            })
    logger.info("chunked paper_id=%s into %d heading-bounded chunks", paper_id, len(chunks))
    return chunks
