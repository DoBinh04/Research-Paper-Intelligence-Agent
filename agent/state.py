"""Agent state definition for LangGraph pipeline."""

from typing import Annotated, Any, Optional
from typing_extensions import TypedDict

import operator


class AgentState(TypedDict, total=False):
    query: str
    session_id: str
    rewritten_query: str
    papers_fetched: list[dict]
    chunks: list[dict]
    retrieved_chunks: list[dict]
    citation_graph: dict
    graph_summary: str
    related_paper_ids: list[str]
    conflicts: list[dict]
    final_answer: str
    report_markdown: str
    loop_count: int
    cost_usd: float
    latency_ms: int
    errors: Annotated[list[str], operator.add]
    status: str
