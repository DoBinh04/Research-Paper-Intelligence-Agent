"""Public LangChain tool for hybrid RAG retrieval."""

from __future__ import annotations

from typing import Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from rag.retrieve import retrieve_chunks


class RAGRetrieveInput(BaseModel):
    query: str
    top_k: int = Field(default=5, le=20)
    paper_id: Optional[str] = None
    section_type: Optional[str] = None
    section_heading: Optional[str] = None


@tool("rag_retrieve", args_schema=RAGRetrieveInput)
def rag_retrieve_tool(
    query: str,
    top_k: int = 5,
    paper_id: Optional[str] = None,
    section_type: Optional[str] = None,
    section_heading: Optional[str] = None,
) -> list[dict]:
    """Retrieve relevant paper chunks using hybrid search.

    This tool exposes the RAG retrieval pipeline to the agent. It performs
    dense Chroma and BM25 retrieval, optionally restricts the search space
    using metadata filters, fuses candidates, and reranks the final list.

    Filtering can be applied to:
    - ``paper_id``: search within a specific paper.
    - ``section_type``: search within a section category (for example,
      ``ABSTRACT``, ``INTRODUCTION``, ``METHOD``, ``RESULTS``,
      ``DISCUSSION``, ``CONCLUSION``, ``THEOREM``, or ``PROOF``).
    - ``section_heading``: search within a specific section heading.

    Args:
        query: Natural language search query describing the information to
            retrieve.
        top_k: Maximum number of ranked chunks to return.
        paper_id: Optional identifier of the paper to search within.
        section_type: Optional section category used to filter results.
            The value is converted to uppercase before searching.
        section_heading: Optional section heading used to filter results.

    Returns:
        A list of dictionaries representing the most relevant document
        chunks. Each dictionary contains the chunk content together with
        metadata such as paper ID, paper title, section information, page
        number, chunk ID, and retrieval score.
    """
    filt = {}
    if paper_id:
        filt["paper_id"] = paper_id
    if section_type:
        filt["section_type"] = section_type.upper()
    if section_heading:
        filt["section_heading"] = section_heading
    return retrieve_chunks(query, top_k=top_k, filter_meta=filt or None)
