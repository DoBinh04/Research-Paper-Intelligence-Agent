"""Serialize NetworkX graph for frontend visualization."""

from __future__ import annotations

import json
from typing import Any

import networkx as nx


def graph_to_json(graph: nx.DiGraph) -> dict[str, Any]:
    """Convert a NetworkX directed graph into a JSON-serializable dictionary.

        The output contains two lists:
            - nodes: Paper metadata (title, year, venue, citation count) together
              with a visualization size derived from the citation count.
            - edges: Directed citation relationships between papers, including
              their hop level.

        This structure is intended for frontend graph visualization libraries.

        Args:
            graph: Directed citation graph to serialize.

        Returns:
            A dictionary with ``nodes`` and ``edges`` fields that can be directly
            serialized to JSON.
    """
    nodes = []
    for node_id, attrs in graph.nodes(data=True):
        citation_count = attrs.get("citation_count") or 0
        nodes.append({
            "id": str(node_id),
            "title": attrs.get("title") or str(node_id),
            "year": attrs.get("year"),
            "venue": attrs.get("venue"),
            "size": max(5, min(50, citation_count // 10 + 5)),
            "citation_count": citation_count,
        })

    edges = [
        {"source": str(u), "target": str(v), "hop_level": d.get("hop_level", 1)}
        for u, v, d in graph.edges(data=True)
    ]

    return {"nodes": nodes, "edges": edges}


def graph_to_json_string(graph: nx.DiGraph) -> str:
    """Serialize a NetworkX directed graph as a JSON string.

        This is a convenience wrapper around :func:`graph_to_json` that converts
        the resulting dictionary into a JSON-formatted string.

        Args:
            graph: Directed citation graph to serialize.

        Returns:
            A JSON string representing the graph, suitable for API responses or
            storage.
    """
    return json.dumps(graph_to_json(graph))
