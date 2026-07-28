"""Citation graph builder via Semantic Scholar references."""

from __future__ import annotations

from typing import Any, Optional

import httpx
import networkx as nx
from langchain_core.tools import tool
from langsmith import traceable
from pydantic import BaseModel, Field

from config.logging import get_logger
from config.settings import settings
from tools.paper_fetch import _get_with_backoff
from tools.rate_limit import SEMANTIC_SCHOLAR_LIMITER

logger = get_logger(__name__)

S2_BASE = "https://api.semanticscholar.org/graph/v1"
S2_FIELDS = "paperId,title,year,venue,citationCount,references"


class CitationGraphInput(BaseModel):
    paper_id: str = Field(..., description="Semantic Scholar paper ID or s2: prefixed id")
    max_hop: int = Field(default=1, le=3)


def _normalize_s2_id(paper_id: str) -> str:
    """
    Normalize the paper ID before send request to Semantic Scholar API.
    Function handle:
    - If ID is "s2:<id>", remove "s2:"
    - If ID is "arvix:<id>", don't remove anytings

    Args:
        paper_id (str): paper ID

    Returns:
        str: normalized paper ID
    """
    if paper_id.startswith("s2:"):
        return paper_id[3:]
    if paper_id.startswith("arxiv:"):
        return paper_id
    return paper_id


async def _fetch_paper_refs(paper_id: str) -> dict[str, Any]:
    """
    Retrieve detailed information about a paper from the Semantic Scholar API.
    This function normalizes the provided paper identifier, sends an
    asynchronous request to the Semantic Scholar Paper API, and returns
    the paper metadata as a JSON dictionary. If a Semantic Scholar API key
    is configured, it is included in the request headers. The function is
    decorated with a retry mechanism that automatically retries failed
    requests up to three times using exponential backoff.

    Args:
        paper_id (str):
            The identifier of the paper to retrieve. This may be a Semantic
            Scholar paper ID, an ID prefixed with ``s2:``, or another
            supported identifier format.

    Returns:
        dict[str, Any]:
            A dictionary containing the paper information requested through
            ``S2_FIELDS``, such as title, authors, publication year,
            references, citations, and other metadata.
    """
    headers = {}
    if settings.semantic_scholar_api_key:
        headers["x-api-key"] = settings.semantic_scholar_api_key

    pid = _normalize_s2_id(paper_id)
    url = f"{S2_BASE}/paper/{pid}"
    params = {"fields": S2_FIELDS}

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await _get_with_backoff(
            client,
            url,
            limiter=SEMANTIC_SCHOLAR_LIMITER,
            params=params,
            headers=headers,
        )
        return resp.json()


@traceable(name="build_citation_graph")
async def build_citation_graph(
    paper_id: str,
    max_hop: int | None = None,
    max_nodes: int | None = None,
) -> tuple[nx.DiGraph, list[str]]:
    """
    Build a citation graph from a given paper using BFS

    The function query the Sematic Scholar Paper API to retrieve the metadata and
references of the input paper, then recursively explores its references until
the maximum search depth (`max_hop`) or node limit (`max_nodes`) is reached

    :argument
        paper_id (str): paper ID
        max_hop (int): max hops
        max_nodes (int): max nodes

    :returns tuple[nx.DiGraph, list[str]]:
    """
    max_hop = max_hop or settings.max_citation_hop
    max_nodes = max_nodes or settings.max_nodes

    graph = nx.DiGraph()
    related_ids: list[str] = [] #List of successfully processed paper IDs
    visited: set[str] = set() #mark approved paper
    queue: list[tuple[str, int]] = [(paper_id, 0)] #queue BFS

    while queue and graph.number_of_nodes() < max_nodes:
        current_id, depth = queue.pop(0)
        norm_id = _normalize_s2_id(current_id)
        if norm_id in visited or depth > max_hop:
            continue
        visited.add(norm_id)

        try:
            data = await _fetch_paper_refs(current_id if current_id.startswith("arxiv:") else norm_id)
        except Exception as exc:
            logger.warning("failed to fetch refs for %s: %s", current_id, exc)
            continue

        node_id = data.get("paperId") or norm_id
        graph.add_node(
            node_id,
            title=data.get("title") or node_id,
            year=data.get("year"),
            venue=data.get("venue"),
            citation_count=data.get("citationCount") or 0,
        )
        related_ids.append(node_id)

        if depth >= max_hop:
            continue

        for ref in data.get("references") or []:
            ref_id = ref.get("paperId")
            if not ref_id:
                continue
            graph.add_node(
                ref_id,
                title=ref.get("title") or ref_id,
                year=ref.get("year"),
                venue=ref.get("venue"),
                citation_count=ref.get("citationCount") or 0,
            )
            graph.add_edge(node_id, ref_id, hop_level=depth + 1)
            if ref_id not in visited and graph.number_of_nodes() < max_nodes:
                queue.append((ref_id, depth + 1))

    logger.info(
        "citation graph: nodes=%d edges=%d depth=%d",
        graph.number_of_nodes(),
        graph.number_of_edges(),
        max_hop,
    )
    return graph, related_ids


@tool("citation_graph", args_schema=CitationGraphInput)
def citation_graph_tool(paper_id: str, max_hop: int = 2) -> dict:
    """Build multi-hop citation graph from a seed paper."""
    import asyncio
    loop = asyncio.new_event_loop()
    graph, related = loop.run_until_complete(build_citation_graph(paper_id, max_hop))
    loop.close()
    return {
        "related_paper_ids": related,
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
    }
