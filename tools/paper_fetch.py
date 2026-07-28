"""
paper_fetch.py — Semantic Scholar + ArXiv dual-source fetcher
Fetches papers, normalises to PaperRecord, supports batch + single.
Traced via LangSmith automatically through LangChain callback.
"""

from __future__ import annotations

import asyncio
import xml.etree.ElementTree as ET
from typing import Any, Literal, Optional
from datetime import datetime

import httpx
from langchain_core.tools import tool
from langsmith import traceable
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential

from config.settings import settings
from config.logging import get_logger
from rag.entity_extractor import extract_entities
from tools.rate_limit import ARXIV_LIMITER, SEMANTIC_SCHOLAR_LIMITER, retry_after_seconds

logger = get_logger(__name__)

# ─── Schema ──────────────────────────────────────────────────────────────────
class PaperRecord(BaseModel):
    paper_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    abstract: str = ""

    authors: list[str] = Field(default_factory=list)

    year: int | None = None
    venue: str | None = None

    citation_count: int = Field(default=0, ge=0)

    references: list[str] = Field(default_factory=list)

    pdf_url: str | None = None

    source: Literal["semantic_scholar", "arxiv", "unknown"] = "unknown"

    arxiv_id: str | None = None
    s2_id: str | None = None

    raw_metadata: dict[str, Any] = Field(default_factory=dict)

class FetchInput(BaseModel):
    query: str = Field(..., description="Search query for ML papers")
    limit: int = Field(default=10, le=50, description="Max papers to fetch")
    source: Literal["semantic_scholar", "arxiv", "both"] = Field(
        default="both",
        description="Paper source"
    )
    year_from: int = Field(
        default=2020,
        ge=1900,
        le=datetime.now().year,
        description="Only fetch papers published from this year"
    )

# ─── Semantic Scholar ─────────────────────────────────────────────────────────

S2_BASE = "https://api.semanticscholar.org/graph/v1"
S2_FIELDS = (
    "paperId,title,abstract,authors,year,venue,"
    "citationCount,references,openAccessPdf,externalIds"
)


def _search_terms(query: str, entities: dict[str, Any] | None) -> list[str]:
    """Return validated entity terms, falling back to the user's query."""
    keywords = (entities or {}).get("paper_keywords")
    if not isinstance(keywords, list):
        return [query]
    terms = [term.strip() for term in keywords if isinstance(term, str) and term.strip()]
    return terms[:5] or [query]


async def _get_with_backoff(
    client: httpx.AsyncClient,
    url: str,
    *,
    limiter: Any,
    **kwargs: Any,
) -> httpx.Response:
    """Make a quota-aware request and respect an upstream 429 response."""
    for attempt in range(4):
        await limiter.wait()
        response = await client.get(url, **kwargs)
        if response.status_code != 429:
            response.raise_for_status()
            return response

        delay = retry_after_seconds(response.headers, fallback=2.0 ** attempt)
        limiter.defer(delay)
        logger.warning("rate limited by %s; retrying in %.2fs", url, delay)

    # Raise the final response with its useful HTTP error details.
    response.raise_for_status()
    raise AssertionError("unreachable")

@traceable(name="fetch_semantic_scholar")
async def _fetch_s2(
    query: str,
    limit: int,
    year_from: Optional[int],
    entities: dict[str, Any] | None = None,
) -> list[PaperRecord]:
    """
    Search for papers using the Semantic Scholar API and convert the
    returned results into a list of PaperRecord objects.

    The function sends a query to the Semantic Scholar Paper Search API,
    retrieves paper metadata, extracts relevant fields such as title,
    abstract, authors, citation count, references, and PDF links, and
    normalizes the data into the application's PaperRecord format.

    If a paper has an ArXiv identifier but does not provide an open-access
    PDF URL, the function automatically constructs the corresponding ArXiv
    PDF link.

        Args:
        query: Search query string.
        limit: Maximum number of papers to retrieve.
        year_from: Minimum publication year (inclusive). If provided,
            only papers published from this year onward are returned.

    Returns:
        A list of PaperRecord objects containing normalized paper
        metadata from Semantic Scholar.
    """
    entities = entities or extract_entities(query)
    search_query = " ".join(_search_terms(query, entities))
    logger.info("semantic_scholar search query: %r", search_query)

    headers = {}
    if settings.semantic_scholar_api_key:
        headers["x-api-key"] = settings.semantic_scholar_api_key

    params: dict[str, Any] = {
        "query": search_query,
        "limit": limit,
        "fields": S2_FIELDS,
    }
    if year_from:
        params["year"] = f"{year_from}-"
    year_hint = entities.get("year_hint")
    if isinstance(year_hint, int) and not isinstance(year_hint, bool):
        params["year"] = f"{year_hint}-{year_hint + 2}"

    async with httpx.AsyncClient(timeout=30) as client: #Create http async client
        resp = await _get_with_backoff(
            client,
            f"{S2_BASE}/paper/search",
            limiter=SEMANTIC_SCHOLAR_LIMITER,
            params=params,
            headers=headers,
        )
        data = resp.json()

    records: list[PaperRecord] = []
    for item in data.get("data", []):
        ref_ids = [
            r["paperId"] for r in (item.get("references") or []) if r.get("paperId")
        ]
        pdf_url = None
        if item.get("openAccessPdf"):
            pdf_url = item["openAccessPdf"].get("url")

        arxiv_id = None
        ext = item.get("externalIds") or {}
        if "ArXiv" in ext:
            arxiv_id = ext["ArXiv"]
            if not pdf_url:
                pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

        records.append(
            PaperRecord(
                paper_id=f"s2:{item['paperId']}",
                title=item.get("title") or "",
                abstract=item.get("abstract") or "",
                authors=[a["name"] for a in (item.get("authors") or [])],
                year=item.get("year"),
                venue=item.get("venue"),
                citation_count=item.get("citationCount") or 0,
                references=ref_ids,
                pdf_url=pdf_url,
                source="semantic_scholar",
                arxiv_id=arxiv_id,
                s2_id=item["paperId"],
                raw_metadata=item,
            )
        )
    logger.info("semantic_scholar returned %d papers for query=%r", len(records), query)
    return records


# ─── ArXiv ───────────────────────────────────────────────────────────────────

ARXIV_BASE = "https://export.arxiv.org/api/query"
ARXIV_NS = "https://www.w3.org/2005/Atom"


@traceable(name="fetch_arxiv")
async def _fetch_arxiv(
    query: str,
    limit: int,
    year_from: Optional[int],
    entities: dict[str, Any] | None = None,
) -> list[PaperRecord]:
    """
    Fetch papers from arXiv and convert the results into a list of
    PaperRecord objects.
    The function queries the arXiv API using the provided search term,
    optionally filters papers by submission year, and restricts results
    to the Machine Learning category (cs.LG). The XML response is parsed
    to extract metadata such as title, abstract, authors, publication
    year, and PDF URL.

    Args:
        query: Search query string.
        limit: Maximum number of papers to retrieve.
        year_from: Minimum submission year (inclusive). If provided,
            only papers submitted from this year onward are returned.

    Returns:
        A list of PaperRecord objects containing normalized metadata
        from arXiv search results.
    """
    terms = _search_terms(query, entities)
    arxiv_terms = [term.replace('"', "").replace("\\", "").strip() for term in terms]
    arxiv_terms = [f'all:"{term}"' for term in arxiv_terms if term]
    search_query = f"({' OR '.join(arxiv_terms)}) AND cat:cs.LG"
    if year_from:
        search_query += f" AND submittedDate:[{year_from}01010000 TO 99991231235959]"

    logger.info("arxiv search query: %r", search_query)

    params = {
        "search_query": search_query,
        "max_results": limit,
        "sortBy": "relevance",
        "sortOrder": "descending",
    }

    async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
        resp = await _get_with_backoff(
            client, ARXIV_BASE, limiter=ARXIV_LIMITER, params=params
        )
        xml_text = resp.text

    root = ET.fromstring(xml_text)
    #define namespace before find node
    ns = {"atom": ARXIV_NS}
    records: list[PaperRecord] = []

    for entry in root.findall("atom:entry", ns): #Each <entry> is a paper
        arxiv_id_raw = (entry.findtext("atom:id", namespaces=ns) or "").split("/abs/")[-1]
        arxiv_id = arxiv_id_raw.strip()
        title = (entry.findtext("atom:title", namespaces=ns) or "").replace("\n", " ").strip()
        abstract = (entry.findtext("atom:summary", namespaces=ns) or "").replace("\n", " ").strip()
        authors = [
            a.findtext("atom:name", namespaces=ns) or ""
            for a in entry.findall("atom:author", ns)
        ]
        published = entry.findtext("atom:published", namespaces=ns) or ""
        year = int(published[:4]) if published else None
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

        records.append(
            PaperRecord(
                paper_id=f"arxiv:{arxiv_id}",
                title=title,
                abstract=abstract,
                authors=authors,
                year=year,
                venue="arXiv",
                pdf_url=pdf_url,
                source="arxiv",
                arxiv_id=arxiv_id,
                raw_metadata={"arxiv_id": arxiv_id, "published": published},
            )
        )

    logger.info("arxiv returned %d papers for query=%r", len(records), query)
    return records


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
async def download_pdf(url: str) -> bytes:
    """
    Download a PDF from a URL and return its binary content.

    The function performs an HTTP GET request, follows redirects, and
    returns the response body as bytes. A warning is logged if the
    response does not appear to be a PDF based on its Content-Type
    header or URL extension.

    Args:
        url: URL of the PDF file.

    Returns:
        Binary content of the downloaded file.
    """
    async with httpx.AsyncClient(timeout=60, follow_redirects=True) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        content_type = resp.headers.get("content-type", "")
        if "pdf" not in content_type.lower() and not url.endswith(".pdf"):
            logger.warning("URL may not be PDF: %s", url)
        return resp.content


# ─── Public tool ─────────────────────────────────────────────────────────────

@traceable(name="paper_fetch_tool")
async def _run_fetch(query: str, limit: int, source: str, year_from: Optional[int]) -> list[PaperRecord]:
    """
     Fetch papers from supported sources and return a
     de-duplicated list of PaperRecord objects.

     Depending on the selected source, the function queries Semantic
     Scholar, arXiv, or both concurrently. When both sources are used,
     the requested limit is split between them. Results from all sources
     are merged, fetch errors are logged without interrupting execution,
     and duplicate papers are removed using the arXiv identifier when
     available, or the paper identifier otherwise.

     Args:
         query: Search query string.
         limit: Maximum number of papers to return.
         source: Data source to query. Supported values are
             "semantic_scholar", "arxiv", and "both".
         year_from: Minimum publication/submission year (inclusive).
             If provided, only papers from this year onward are fetched.

     Returns:
         A list of unique PaperRecord objects collected from the selected
         source(s), limited to the requested number of results.
     """
    tasks = []
    half = max(1, limit // 2)
    entities = extract_entities(query)

    if source in ("semantic_scholar", "both"):
        tasks.append(
            _fetch_s2(query, limit if source != "both" else half, year_from, entities)
        )
    if source in ("arxiv", "both"):
        tasks.append(
            _fetch_arxiv(query, limit if source != "both" else half, year_from, entities)
        )

    results = await asyncio.gather(*tasks, return_exceptions=True)
    papers: list[PaperRecord] = []
    for r in results:
        if isinstance(r, Exception):
            logger.error("Fetch error: %s", r)
        else:
            papers.extend(r)

    # De-duplicate by arxiv_id
    seen: set[str] = set()
    unique: list[PaperRecord] = []
    for p in papers:
        key = p.arxiv_id or p.paper_id
        if key not in seen:
            seen.add(key)
            unique.append(p)

    logger.info("Total unique papers fetched: %d", len(unique))
    return unique[:limit]


@tool("paper_fetch", args_schema=FetchInput)
def paper_fetch_tool(
    query: str,
    limit: int = 10,
    source: str = "both",
    year_from: Optional[int] = 2020,
) -> list[dict[str, Any]]:
    """
    Fetch research papers from Semantic Scholar and/or arXiv.

    The function retrieves papers from the selected source,
    converts the resulting PaperRecord objects into dictionaries,
    and returns normalized metadata suitable for downstream
    processing and ingestion.

    Args:
        query: Search query string.
        limit: Maximum number of papers to return.
        source: Source to query ("semantic_scholar", "arxiv", or "both").
        year_from: Minimum publication/submission year (inclusive).

    Returns:
        A list of dictionaries containing normalized paper metadata.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        papers = loop.run_until_complete(
            _run_fetch(query, limit, source, year_from)
        )
    finally:
        loop.close()

    return [p.model_dump(mode="json") for p in papers]

# ─── CLI helper ──────────────────────────────────────────────────────────────

# async def fetch_50_ml_papers() -> list[PaperRecord]:
#     """Convenience: fetch 50 ML papers across multiple sub-topics."""
#     queries = [
#         "large language model reasoning",
#         "diffusion model image generation",
#         "reinforcement learning from human feedback",
#         "graph neural network",
#         "vision transformer self-supervised",
#     ]
#     all_papers: list[PaperRecord] = []
#     for q in queries:
#         papers = await _run_fetch(q, limit=10, source="both", year_from=2021)
#         all_papers.extend(papers)
#         time.sleep(1)  # polite rate-limit
#
#     # Final dedup
#     seen: set[str] = set()
#     unique: list[PaperRecord] = []
#     for p in all_papers:
#         key = p.arxiv_id or p.paper_id
#         if key not in seen:
#             seen.add(key)
#             unique.append(p)
#
#     logger.info("fetch_50_ml_papers: collected %d unique papers", len(unique))
#     return unique[:50]
#
#
# if __name__ == "__main__":
#     import json
#     papers = asyncio.run(fetch_50_ml_papers())
#     print(json.dumps([p.model_dump() for p in papers[:3]], indent=2))
