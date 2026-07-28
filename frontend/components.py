"""Reusable Streamlit UI components."""

import streamlit as st


def render_metrics(cost: float, latency_ms: int) -> None:
    col1, col2 = st.columns(2)
    col1.metric("Cost (USD)", f"${cost:.4f}")
    col2.metric("Latency", f"{latency_ms} ms")


def render_report(markdown: str) -> None:
    st.markdown(markdown)


def render_citation_graph(nodes: list[dict], edges: list[dict]) -> None:
    if not nodes:
        st.info("No citation graph data.")
        return

    try:
        from streamlit_agraph import agraph, Node, Edge, Config

        graph_nodes = [
            Node(
                id=n["id"],
                label=(n.get("title") or n["id"])[:40],
                size=n.get("size", 15),
                title=f"{n.get('title')} ({n.get('year')})",
            )
            for n in nodes[:30]
        ]
        node_ids = {n.id for n in graph_nodes}
        graph_edges = [
            Edge(source=e["source"], target=e["target"])
            for e in edges
            if e["source"] in node_ids and e["target"] in node_ids
        ]
        config = Config(
            width=900,
            height=500,
            directed=True,
            nodeHighlightBehavior=True,
            highlightColor="#F7A7A6",
            collapsible=False,
        )
        agraph(nodes=graph_nodes, edges=graph_edges, config=config)
    except ImportError:
        st.json({"nodes": nodes[:10], "edges": edges[:20]})
