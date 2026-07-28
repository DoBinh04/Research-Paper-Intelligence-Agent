"""Pydantic schemas for FastAPI."""

from typing import Optional

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    session_id: Optional[str] = None


class QueryResponse(BaseModel):
    answer: str
    report_md: str
    cost: float = 0.0
    latency_ms: int = 0
    session_id: str = ""
    conflicts: list[dict] = Field(default_factory=list)
    graph: dict = Field(default_factory=dict)


class IngestRequest(BaseModel):
    paper_id: Optional[str] = None
    query: Optional[str] = None
    limit: int = 3


class IngestResponse(BaseModel):
    status: str
    results: list[dict] = Field(default_factory=list)


class GraphResponse(BaseModel):
    nodes: list[dict]
    edges: list[dict]
    stats: dict = Field(default_factory=dict)
