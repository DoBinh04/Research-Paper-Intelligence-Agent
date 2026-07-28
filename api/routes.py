"""FastAPI routes."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException

from agent.graph import run_agent
from api.schemas import GraphResponse, IngestRequest, IngestResponse, QueryRequest, QueryResponse
from config.logging import get_logger
from graph_engine.traversal import traverse_citation_graph
from graph_engine.visualizer import graph_to_json
from rag.pipeline import ingest_by_paper_id, ingest_from_query

logger = get_logger(__name__)
router = APIRouter()


@router.post("/query", response_model=QueryResponse)
async def query_papers(body: QueryRequest) -> QueryResponse:
    if not body.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    try:
        result = await asyncio.to_thread(run_agent, body.query, body.session_id)
    except Exception as exc:
        logger.exception("agent failed")
        raise HTTPException(status_code=503, detail=f"Agent error: {exc}") from exc

    return QueryResponse(
        answer=result.get("final_answer") or "",
        report_md=result.get("report_markdown") or "",
        cost=float(result.get("cost_usd") or 0.0),
        latency_ms=int(result.get("latency_ms") or 0),
        session_id=result.get("session_id") or "",
        conflicts=result.get("conflicts") or [],
        graph=result.get("citation_graph") or {},
    )


@router.post("/ingest", response_model=IngestResponse)
async def ingest_papers(body: IngestRequest) -> IngestResponse:
    try:
        if body.paper_id:
            result = await ingest_by_paper_id(body.paper_id)
            return IngestResponse(status="ok", results=[result])
        if body.query:
            results = await ingest_from_query(body.query, limit=body.limit)
            return IngestResponse(status="ok", results=results)
        raise HTTPException(status_code=400, detail="Provide paper_id or query")
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("ingest failed")
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/graph/{paper_id}", response_model=GraphResponse)
async def get_citation_graph(paper_id: str, max_hop: int = 2) -> GraphResponse:
    try:
        result = await traverse_citation_graph(paper_id, max_depth=max_hop)
        graph_json = graph_to_json(result["graph"])
        return GraphResponse(
            nodes=graph_json["nodes"],
            edges=graph_json["edges"],
            stats=result["stats"],
        )
    except Exception as exc:
        logger.exception("graph fetch failed")
        raise HTTPException(status_code=404, detail=f"Paper not found or API error: {exc}") from exc
