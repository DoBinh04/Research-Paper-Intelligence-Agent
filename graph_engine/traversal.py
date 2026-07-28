"""BFS citation graph traversal."""

from __future__ import annotations

from typing import Any

import networkx as nx

from config.logging import get_logger
from config.settings import settings
from tools.citation_graph import build_citation_graph

logger = get_logger(__name__)


async def traverse_citation_graph(
    seed_paper_id: str,
    max_depth: int | None = None,
    max_nodes: int | None = None,
) -> dict[str, Any]:
    """Traverse the citation graph starting from a seed paper.

    Builds a citation graph using breadth-first search (BFS) from the
    specified seed paper, then computes basic traversal statistics such as
    the number of discovered papers, citation edges, and the maximum hop
    level reached during traversal.

    Args:
        seed_paper_id: Semantic Scholar paper ID of the starting paper.
        max_depth: Maximum BFS depth. If ``None``, uses the default value
            from the application settings.
        max_nodes: Maximum number of papers to include in the graph. If
            ``None``, uses the default value from the application settings.

    Returns:
        A dictionary containing:
            - ``graph``: The constructed citation graph.
            - ``related_ids``: IDs of all discovered papers.
            - ``stats``: Traversal statistics including the number of nodes,
              edges, and maximum depth reached.
    """

    max_depth = max_depth or settings.max_citation_hop
    max_nodes = max_nodes or settings.max_nodes

    graph, related_ids = await build_citation_graph(
        seed_paper_id,
        max_hop=max_depth,
        max_nodes=max_nodes,
    )

    depth_reached = 0
    if graph.number_of_edges() > 0:
        for _, _, data in graph.edges(data=True):
            depth_reached = max(depth_reached, data.get("hop_level", 1))

    stats = {
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "depth_reached": depth_reached,
    }
    logger.info("graph traversal stats: %s", stats)
    return {"graph": graph, "related_ids": related_ids, "stats": stats}


def summarize_graph(graph: nx.DiGraph, limit: int = 10) -> str:
    """Generate a human-readable summary of a citation graph.

        Papers are ranked by citation count in descending order, and only the
        top ``limit`` papers are included in the summary.

        Args:
            graph: Citation graph represented as a directed NetworkX graph.
            limit: Maximum number of papers to include in the summary.

        Returns:
            A formatted string describing the graph size and listing the most
            highly cited papers. Returns ``"Empty citation graph."`` if the
            graph contains no nodes.
    """

    if graph.number_of_nodes() == 0:
        return "Empty citation graph."

    nodes = sorted(
        graph.nodes(data=True),
        key=lambda x: x[1].get("citation_count", 0),
        reverse=True,
    )[:limit]

    lines = [f"Citation graph: {graph.number_of_nodes()} papers, {graph.number_of_edges()} cite edges."]
    for node_id, attrs in nodes:
        lines.append(
            f"- {attrs.get('title', node_id)} ({attrs.get('year', '?')}), "
            f"citations={attrs.get('citation_count', 0)}"
        )
    return "\n".join(lines)
