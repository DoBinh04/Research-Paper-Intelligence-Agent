"""LangGraph orchestrator: fetch -> parse -> retrieve -> graph -> synthesize."""

from __future__ import annotations

import asyncio
import time
import uuid
from collections import Counter
from typing import Any

from langgraph.graph import END, StateGraph

from agent.memory import get_memory
from agent.state import AgentState
from config.logging import get_logger
from graph_engine.conflict_detector import detect_conflicts
from graph_engine.traversal import summarize_graph, traverse_citation_graph
from graph_engine.visualizer import graph_to_json
from rag.pipeline import ingest_paper_record
from rag.query_rewriter import rewrite_query
from tools.paper_fetch import PaperRecord, _run_fetch
from tools.rag_retrieve import retrieve_chunks
from tools.report_gen import generate_report

logger = get_logger(__name__)


def _select_retrieval_seed(
    papers: list[dict[str, Any]], chunks: list[dict[str, Any]]
) -> tuple[dict[str, Any], int]:
    """Select the fetched paper represented by the most retrieved chunks.

    Only paper IDs in the current fetch batch are counted, so chunks left in
    the persistent RAG collection from older queries cannot become the seed.
    Ties retain fetch order for deterministic behaviour.
    """
    contribution_counts = Counter(
        str(chunk.get("paper_id"))
        for chunk in chunks
        if chunk.get("paper_id")
    )
    seed = max(
        papers,
        key=lambda paper: contribution_counts.get(str(paper.get("paper_id", "")), 0),
    )
    return seed, contribution_counts.get(str(seed.get("paper_id", "")), 0)


async def _fetch_node(state: AgentState) -> dict[str, Any]:
    """Fetch candidate papers for the current query.

        The node first checks the session memory for previously fetched papers.
        If cached papers exist and this is the first graph iteration, the cached
        results are reused to avoid unnecessary API requests. Otherwise, papers
        are fetched from the configured sources, stored in the session memory,
        and returned for downstream processing.

        Args:
            state: Current LangGraph state containing the user query, session
                identifier, and loop information.

        Returns:
            State updates containing the fetched papers and the fetch status.
    """
    query = state["query"]
    memory = get_memory()
    session_id = state.get("session_id") or "default"
    memory.add_query(session_id, query)

    cached = memory.list_papers(session_id)
    if cached and state.get("loop_count", 0) == 0:
        logger.info("using %d cached papers for session=%s", len(cached), session_id)
        return {"papers_fetched": cached, "status": "fetch_cached"}

    papers = await _run_fetch(query, limit=3, source="both", year_from=None)
    paper_dicts = [p.model_dump() for p in papers]
    for p in paper_dicts:
        memory.set_paper(session_id, p["paper_id"], p)

    return {"papers_fetched": paper_dicts, "status": "fetch_done"}


async def _parse_ingest_node(state: AgentState) -> dict[str, Any]:
    """Ingest fetched papers into the RAG index and retrieve relevant chunks.

        Each fetched paper is parsed, chunked, embedded, and stored in the
        retrieval index through the ingest pipeline. After ingestion, the user
        query is rewritten for academic retrieval, and the most relevant chunks
        are retrieved for downstream reasoning.

        Args:
            state: Current LangGraph state containing the fetched papers and
                original user query.

        Returns:
            State updates containing the rewritten query, retrieved chunks,
            aggregated chunk list, and ingestion status.
    """
    papers = state.get("papers_fetched") or []
    all_chunks: list[dict] = []

    for paper_dict in papers:
        paper = PaperRecord(**paper_dict)
        result = await ingest_paper_record(paper)
        logger.info("ingest %s -> %s", paper.paper_id, result.get("status"))

    rewritten = rewrite_query(state["query"])
    chunks = retrieve_chunks(rewritten, top_k=10, rewrite=False)
    all_chunks.extend(chunks)

    return {
        "rewritten_query": rewritten,
        "retrieved_chunks": chunks,
        "chunks": all_chunks,
        "status": "ingest_done",
    }


async def _graph_node(state: AgentState) -> dict[str, Any]:
    papers = state.get("papers_fetched") or []
    if not papers:
        return {"graph_summary": "No papers to trace.", "citation_graph": {"nodes": [], "edges": []}}

    seed, matched_chunks = _select_retrieval_seed(
        papers, state.get("retrieved_chunks") or state.get("chunks") or []
    )
    seed_id = seed.get("s2_id") or seed.get("paper_id", "").replace("s2:", "")
    logger.info(
        "citation graph seed=%s retrieved_chunk_count=%d",
        seed.get("paper_id"),
        matched_chunks,
    )

    try:
        result = await traverse_citation_graph(seed_id)
        graph = result["graph"]
        summary = summarize_graph(graph)
        graph_json = graph_to_json(graph)
        return {
            "citation_graph": graph_json,
            "graph_summary": summary,
            "related_paper_ids": result["related_ids"],
            "status": "graph_done",
        }
    except Exception as exc:
        logger.error("graph trace failed: %s", exc)
        return {
            "graph_summary": f"Citation graph unavailable: {exc}",
            "citation_graph": {"nodes": [], "edges": []},
            "errors": [str(exc)],
            "status": "graph_failed",
        }


async def _synthesize_node(state: AgentState) -> dict[str, Any]:
    """
        Generate the final literature review report from retrieved evidence.

        This is the final node in the LangGraph workflow. It gathers the
        retrieved document chunks, detects conflicting experimental findings,
        and invokes the report generator to synthesize a coherent answer.

        Workflow:
            1. Retrieve the relevant chunks from the current state.
            2. Detect conflicting numerical or experimental claims.
            3. Generate a final report using:
                - the user's query,
                - retrieved evidence,
                - citation graph summary,
                - detected conflicts.
            4. Store the generated answer and Markdown report back into
               the workflow state.

        Args:
            state: Shared workflow state containing retrieved chunks,
                graph summary, and the user's research query.

        Returns:
            A dictionary containing:
                - conflicts: Detected conflicting claims.
                - final_answer: Concise answer generated by the LLM.
                - report_markdown: Full literature review in Markdown format.
                - status: Workflow status ("complete").
        """
    chunks = state.get("retrieved_chunks") or state.get("chunks") or []
    conflicts = detect_conflicts(chunks, topic=state.get("query", ""))

    report = generate_report(
        query=state["query"],
        chunks=chunks,
        graph_summary=state.get("graph_summary", ""),
        conflicts=conflicts,
    )

    return {
        "conflicts": conflicts,
        "final_answer": report["answer"],
        "report_markdown": report["markdown_report"],
        "status": "complete",
    }


def build_graph() -> StateGraph:
    workflow = StateGraph(AgentState)

    workflow.add_node("fetch", lambda s: asyncio.run(_fetch_node(s)))
    workflow.add_node("parse_ingest", lambda s: asyncio.run(_parse_ingest_node(s)))
    workflow.add_node("graph_trace", lambda s: asyncio.run(_graph_node(s)))
    workflow.add_node("synthesize", lambda s: asyncio.run(_synthesize_node(s)))

    workflow.set_entry_point("fetch")
    workflow.add_edge("fetch", "parse_ingest")
    workflow.add_edge("parse_ingest", "graph_trace")
    workflow.add_edge("graph_trace", "synthesize")
    workflow.add_edge("synthesize", END)

    return workflow.compile()


_compiled_graph = None


def get_agent_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


def run_agent(query: str, session_id: str | None = None) -> AgentState:
    start = time.perf_counter()
    session_id = session_id or str(uuid.uuid4())

    initial: AgentState = {
        "query": query,
        "session_id": session_id,
        "loop_count": 0,
        "cost_usd": 0.0,
        "errors": [],
    }

    graph = get_agent_graph()
    result = graph.invoke(initial)
    result["latency_ms"] = int((time.perf_counter() - start) * 1000)
    logger.info("agent completed in %dms session=%s", result["latency_ms"], session_id)
    return result
