"""Streamlit demo UI for Research Paper Intelligence Agent."""

import uuid

import httpx
import streamlit as st

from config.settings import settings
from frontend.components import render_citation_graph, render_metrics, render_report

API_BASE = settings.api_base_url

st.set_page_config(
    page_title="Research Paper Intelligence Agent",
    page_icon="📄",
    layout="wide",
)

st.title("Research Paper Intelligence Agent")
st.caption("Fetch → Parse → Citation Graph → RAG → Synthesize")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

tab_chat, tab_report, tab_graph, tab_ingest = st.tabs(["Chat", "Report", "Graph", "Ingest"])

with tab_chat:
    query = st.text_input(
        "Research question",
        placeholder='e.g. "What is ProLoRA"',
    )
    if st.button("Run Agent", type="primary") and query:
        with st.spinner("Running multi-step agent pipeline..."):
            try:
                resp = httpx.post(
                    f"{API_BASE}/query",
                    json={"query": query, "session_id": st.session_state.session_id},
                    timeout=300,
                )
                resp.raise_for_status()
                data = resp.json()
                st.session_state.last_result = data
            except httpx.HTTPError as exc:
                st.error(f"API error: {exc}")

    if result := st.session_state.get("last_result"):
        st.subheader("Answer")
        st.write(result.get("answer", ""))
        render_metrics(result.get("cost", 0), result.get("latency_ms", 0))
        if conflicts := result.get("conflicts"):
            st.subheader("Detected Conflicts")
            st.json(conflicts)

with tab_report:
    if result := st.session_state.get("last_result"):
        report = result.get("report_md", "")
        render_report(report)
        st.download_button("Export Markdown", report, file_name="research_report.md")
    else:
        st.info("Run a query in the Chat tab first.")

with tab_graph:
    paper_id = st.text_input("Paper ID (Semantic Scholar or s2:...)", value="")
    max_hop = st.slider("Max hops", 1, 3, 2)
    if st.button("Load Graph") and paper_id:
        try:
            resp = httpx.get(f"{API_BASE}/graph/{paper_id}", params={"max_hop": max_hop}, timeout=120)
            resp.raise_for_status()
            graph_data = resp.json()
            st.session_state.graph_data = graph_data
        except httpx.HTTPError as exc:
            st.error(f"Graph error: {exc}")

    if graph_data := st.session_state.get("graph_data"):
        st.json(graph_data.get("stats", {}))
        render_citation_graph(graph_data.get("nodes", []), graph_data.get("edges", []))
    elif result := st.session_state.get("last_result"):
        graph = result.get("graph", {})
        render_citation_graph(graph.get("nodes", []), graph.get("edges", []))

with tab_ingest:
    ingest_query = st.text_input("Ingest by search query", placeholder="attention is all you need")
    ingest_limit = st.number_input("Limit", 1, 10, 3)
    if st.button("Ingest Papers") and ingest_query:
        try:
            resp = httpx.post(
                f"{API_BASE}/ingest",
                json={"query": ingest_query, "limit": ingest_limit},
                timeout=300,
            )
            resp.raise_for_status()
            st.json(resp.json())
        except httpx.HTTPError as exc:
            st.error(f"Ingest error: {exc}")
