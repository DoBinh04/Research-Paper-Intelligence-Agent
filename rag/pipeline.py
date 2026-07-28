"""Ingest pipeline: parse PDF blocks -> chunk -> embed -> ChromaDB."""

from __future__ import annotations

import argparse
import asyncio
import json
from typing import Any

from config.logging import get_logger
from rag.chunker import chunk_blocks
from rag.bm25_retriever import get_bm25_retriever
from rag.embedder import embed_texts
from rag.store import get_chunks_collection, get_meta_collection, paper_exists
from tools.math_parser import parse_pdf
from tools.paper_fetch import PaperRecord, _run_fetch, download_pdf

logger = get_logger(__name__)


def ingest_blocks(blocks: list[dict], paper_id: str, metadata: dict) -> dict[str, Any]:
    """Ingest parsed paper blocks into ChromaDB after regenerating chunks.

    If the paper already exists, all previously stored chunks and metadata are
    removed so that changes in the parser or chunking strategy are reflected in
    the database. The function then chunks the parsed blocks, generates
    embeddings for each chunk, stores the chunk documents together with their
    metadata, and finally stores paper-level metadata in a separate collection.

    Args:
        blocks: Ordered content blocks produced by the PDF parser. Each block
            contains the extracted content and structural information such as
            heading hierarchy, page number, and block type.
        paper_id: Unique identifier of the paper.
        metadata: Paper-level metadata, such as title, publication year,
            venue, and citation count.

    Returns:
        A dictionary describing the ingestion result containing:

        - ``status``: ``"ok"`` if ingestion succeeds or ``"empty"`` if no
          chunks were generated.
        - ``chunks_stored``: Number of chunks stored in the vector database.
        - ``paper_id``: Identifier of the ingested paper.
    """
    if paper_exists(paper_id):
        get_chunks_collection().delete(where={"paper_id": paper_id})
        get_meta_collection().delete(ids=[paper_id])
        logger.info("deleted existing chunks for paper_id=%s before re-ingest", paper_id)

    title = metadata.get("title") or paper_id
    chunks = chunk_blocks(blocks, paper_id, paper_title=title)
    if not chunks:
        get_bm25_retriever().replace_paper(paper_id, [])
        return {"status": "empty", "chunks_stored": 0, "paper_id": paper_id}

    texts = [c["content"] for c in chunks]
    vectors = embed_texts(texts)

    coll = get_chunks_collection()
    ids = [f"{paper_id}:{c['chunk_id']}" for c in chunks]
    metadatas = [
        {
            "paper_id": paper_id,
            "paper_title": c["paper_title"],
            "section_type": c["section_type"],
            "section_heading": c["section_heading"],
            "subsection_heading": c["subsection_heading"],
            "heading_path": c["heading_path"],
            "chunk_index": int(c["chunk_index"]),
            "heading_chunk_index": int(c["heading_chunk_index"]),
            "chunk_id": c["chunk_id"],
            "page": int(c.get("page") or 0),
            "is_math": bool(c.get("is_math")),
            "is_table": bool(c.get("is_table")),
        }
        for c in chunks
    ]

    coll.add(ids=ids, documents=texts, embeddings=vectors, metadatas=metadatas)

    bm25_chunks = [
        {
            "content": text, "chunk_id": metadata["chunk_id"],
            "paper_id": metadata["paper_id"], "paper_title": metadata["paper_title"],
            "section_type": metadata["section_type"], "section_heading": metadata["section_heading"],
            "subsection_heading": metadata["subsection_heading"], "heading_path": metadata["heading_path"],
            "chunk_index": metadata["chunk_index"], "page": metadata["page"],
        }
        for text, metadata in zip(texts, metadatas, strict=True)
    ]
    get_bm25_retriever().replace_paper(paper_id, bm25_chunks)

    meta_coll = get_meta_collection()
    meta_coll.add(
        ids=[paper_id],
        documents=[metadata.get("title", paper_id)],
        metadatas=[{
            "paper_id": paper_id,
            "title": metadata.get("title", ""),
            "year": metadata.get("year") or 0,
            "venue": metadata.get("venue") or "",
            "citation_count": metadata.get("citation_count") or 0,
        }],
    )

    logger.info("ingested paper_id=%s chunks=%d", paper_id, len(chunks))
    return {"status": "ok", "chunks_stored": len(chunks), "paper_id": paper_id}


async def ingest_from_query(query: str, limit: int = 3) -> list[dict]:
    """Search, download, and ingest multiple papers matching a query.

    The function retrieves papers from the configured search sources, ingests
    each paper into the vector database, and returns the ingestion result for
    every processed paper.

    Args:
        query: Search query used to retrieve relevant papers.
        limit: Maximum number of papers to ingest.

    Returns:
        A list of ingestion result dictionaries, one for each processed paper.
        Each dictionary contains the status of the ingestion, the paper ID, and
        the number of chunks stored.
    """
    papers = await _run_fetch(query, limit=limit, source="both", year_from=None)
    results = []
    for paper in papers:
        result = await ingest_paper_record(paper)
        results.append(result)
    return results


async def ingest_paper_record(paper: PaperRecord) -> dict[str, Any]:
    """Download, parse, and ingest a single paper into the vector database.

    If a PDF URL is available and the download succeeds, the paper is parsed
    into structured content blocks. Otherwise, a fallback block is created from
    the paper's abstract or title so the paper can still be indexed. The parsed
    blocks and paper metadata are then passed to the ingestion pipeline.

    Args:
        paper: Metadata describing the paper to ingest, including its
            identifier, title, abstract, and optional PDF URL.

    Returns:
        A dictionary describing the ingestion result, including the ingestion
        status, paper identifier, and number of chunks stored.
    """
    pdf_bytes = None
    if paper.pdf_url:
        pdf_bytes = await download_pdf(paper.pdf_url)

    if pdf_bytes:
        blocks = parse_pdf(pdf_bytes)
    else:
        blocks = [{
            "content": paper.abstract or paper.title,
            "section_type": "ABSTRACT",
            "page_num": 0,
            "is_math": False,
            "is_table": False,
            "is_heading": False,
            "heading": "Abstract",
            "heading_level": 1,
            "heading_path": ["Abstract"],
        }]

    metadata = paper.model_dump()
    return ingest_blocks(blocks, paper.paper_id, metadata)


async def ingest_by_paper_id(paper_id: str) -> dict[str, Any]:
    """Retrieve and ingest a paper identified by its paper ID.

    The function searches for the paper using its identifier, which may correspond to a Semantic Scholar ID (``s2:``) or
    an arXiv ID (``arxiv:``). After fetching candidate papers, it selects the paper matching the provided identifier
    when possible and then ingests the paper into the vector database through the ingestion pipeline.

    Args:
        paper_id: Unique identifier of the paper. Supported formats include Semantic Scholar IDs (e.g., ``s2:123456``)
        and arXiv IDs (e.g., ``arxiv:2401.12345``).

    Returns:
        A dictionary describing the ingestion result.
        The dictionary contains ingestion metadata such as status, paper identifier, and the number of chunks stored.
        If no matching paper is found, the returned status is ``"not_found"`` and no chunks are stored. """
    query = paper_id.replace("s2:", "").replace("arxiv:", "")
    papers = await _run_fetch(query, limit=1, source="both", year_from=None)
    if not papers:
        return {"status": "not_found", "chunks_stored": 0, "paper_id": paper_id}
    target = papers[0]
    if paper_id.startswith("s2:") or paper_id.startswith("arxiv:"):
        for p in papers:
            if p.paper_id == paper_id:
                target = p
                break
    return await ingest_paper_record(target)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest papers into ChromaDB")
    parser.add_argument("--query", type=str, help="Search query to fetch papers")
    parser.add_argument("--paper_id", type=str, help="Specific paper id")
    parser.add_argument("--limit", type=int, default=3)
    args = parser.parse_args()

    if args.paper_id:
        result = asyncio.run(ingest_by_paper_id(args.paper_id))
    elif args.query:
        result = asyncio.run(ingest_from_query(args.query, limit=args.limit))
    else:
        parser.error("Provide --query or --paper_id")

    print(json.dumps(result if isinstance(result, dict) else result, indent=2, default=str))


if __name__ == "__main__":
    main()
