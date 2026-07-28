"""
Generate the final research report for the RAG pipeline.

Responsibilities
----------------
1. Collect retrieved document chunks.
2. Format the retrieval context for the LLM.
3. Summarize the evidence into a coherent answer.
4. Include citation graph information.
5. Highlight conflicting findings between papers.
6. Export both:
   - a short answer for the agent response
   - a complete Markdown research report.

If no OpenAI API key is available, the module falls back to a simple
extractive summary so the pipeline can still run.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from agent.prompts import SYNTHESIZE_PROMPT
from config.logging import get_logger
from config.settings import settings

logger = get_logger(__name__)


class ReportInput(BaseModel):
    query: str
    chunks_json: str = Field(..., description="JSON string of retrieved chunks")
    graph_summary: str = ""
    conflicts_json: str = "[]"


def _format_context(chunks: list[dict]) -> str:
    """
    Convert retrieved document chunks into a formatted text block for the LLM.

    Each chunk is assigned a human-readable index (starting from 1) and
    includes its paper title, heading hierarchy, chunk index, page number, and content. The
    formatted chunks are concatenated into a single string, which is used
    as the retrieval context in the synthesis prompt.

    Args:
        chunks: List of retrieved document chunks. Each chunk is expected
            to contain the keys:
            - paper_id
            - paper_title
            - heading_path
            - chunk_index
            - page
            - content

    Returns:
        A single formatted string containing all retrieved chunks,
        separated by blank lines.
    """
    parts = []
    for i, c in enumerate(chunks, 1):
        parts.append(
            f"[{i}] paper={c.get('paper_title') or c.get('paper_id','?')} "
            f"heading={c.get('heading_path') or c.get('section_heading') or c.get('section_type','?')} "
            f"chunk={c.get('chunk_index','?')} page={c.get('page','?')}\n{c.get('content','')}"
        )
    return "\n\n".join(parts)


def _format_conflicts(conflicts: list[dict]) -> str:
    """
    Format detected conflicts into a human-readable text block for the LLM.

    Each conflict describes a disagreement between two papers on the same
    evaluation metric (e.g., Accuracy, F1-score, BLEU). The formatted
    string is included in the synthesis prompt so the LLM can mention or
    analyze conflicting findings in the final report.

    If no conflicts are detected, a default message is returned.

    Args:
        conflicts: List of detected conflicts. Each conflict is expected
            to contain the keys:
            - paper_a
            - claim_a
            - paper_b
            - claim_b
            - metric

    Returns:
        A formatted string describing all detected conflicts, or a default
        message if no conflicts exist.
    """
    if not conflicts:
        return "No numeric conflicts detected."
    lines = []
    for c in conflicts:
        lines.append(
            f"- {c.get('paper_a')} claims {c.get('claim_a')} vs "
            f"{c.get('paper_b')} claims {c.get('claim_b')} on {c.get('metric')}"
        )
    return "\n".join(lines)


def generate_report(
    query: str,
    chunks: list[dict],
    graph_summary: str = "",
    conflicts: list[dict] | None = None,
) -> dict[str, str]:
    conflicts = conflicts or []
    context = _format_context(chunks)
    conflict_text = _format_conflicts(conflicts)

    if not settings.openai_api_key:
        answer = (
            f"Based on {len(chunks)} retrieved sections, here is a summary for: {query}\n\n"
            + context[:2000]
        )
        markdown = (
            f"# Research Report\n\n## Query\n{query}\n\n## Summary\n{answer}\n\n"
            f"## Citation Graph\n{graph_summary or 'N/A'}\n\n## Conflicts\n{conflict_text}\n"
        )
        return {"answer": answer[:500], "markdown_report": markdown}

    llm = ChatOpenAI(
        model=settings.synthesis_model,
        temperature=0,
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
    )
    prompt = SYNTHESIZE_PROMPT.format(
        query=query,
        context=context,
        graph_summary=graph_summary or "No graph data.",
        conflicts=conflict_text,
    )
    response = llm.invoke([
        SystemMessage(content="You are an expert ML research analyst."),
        HumanMessage(content=prompt),
    ])
    content = response.content or ""

    if "## Summary" in content:
        markdown = content
        answer = content.split("## Summary")[-1].split("##")[0].strip()[:800]
    else:
        answer = content[:800]
        markdown = (
            f"# Research Report\n\n## Query\n{query}\n\n## Summary\n{content}\n\n"
            f"## Citation Graph\n{graph_summary}\n\n## Conflicts\n{conflict_text}\n"
        )

    logger.info("generated report for query=%r", query[:60])
    return {"answer": answer, "markdown_report": markdown}


@tool("report_generator", args_schema= ReportInput)
def report_generator_tool(query: str, chunks_json: str, graph_summary: str = "", conflicts_json: str = "[]") -> dict:
    """Generate markdown research report from retrieved chunks."""
    import json

    chunks = json.loads(chunks_json) if chunks_json else []
    conflicts = json.loads(conflicts_json) if conflicts_json else []
    return generate_report(query, chunks, graph_summary, conflicts)
