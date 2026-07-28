"""Extract structured academic search entities from user queries."""

from __future__ import annotations

import json
from typing import Any

from openai import OpenAI

from agent.prompts import ENTITY_EXTRACTION_PROMPT
from config.logging import get_logger
from config.settings import settings

logger = get_logger(__name__)

_FALLBACK = {
    "paper_keywords": [],
    "task_type": "find_paper",
    "domain_hint": None,
    "year_hint": None,
}


def _normalise_entities(value: dict[str, Any], query: str) -> dict[str, Any]:
    """Make an LLM response safe and predictable for the search clients."""
    keywords = value.get("paper_keywords")
    if not isinstance(keywords, list):
        keywords = []
    keywords = [
        keyword.strip()
        for keyword in keywords
        if isinstance(keyword, str) and keyword.strip()
    ][:6]

    task_type = value.get("task_type")
    if task_type not in {"find_paper", "compare", "summarize", "explain"}:
        task_type = "find_paper"

    domain_hint = value.get("domain_hint")
    if not isinstance(domain_hint, str) or not domain_hint.strip():
        domain_hint = None

    year_hint = value.get("year_hint")
    if isinstance(year_hint, bool) or not isinstance(year_hint, int):
        year_hint = None

    return {
        "paper_keywords": keywords or [query],
        "task_type": task_type,
        "domain_hint": domain_hint,
        "year_hint": year_hint,
    }


def _fallback(query: str) -> dict[str, Any]:
    """Return a safe search payload when entity extraction is unavailable."""
    return {**_FALLBACK, "paper_keywords": [query]}


def extract_entities(query: str) -> dict[str, Any]:
    """Use the inexpensive OpenAI model to turn a query into search entities."""
    try:
        client = OpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url)
        response = client.chat.completions.create(
            model=settings.LLM_CHEAP_MODEL,
            messages=[
                {"role": "system", "content": ENTITY_EXTRACTION_PROMPT},
                {"role": "user", "content": query},
            ],
            temperature=0,
        )
        content = response.choices[0].message.content
        if not content:
            raise ValueError("OpenAI returned an empty entity extraction response")

        try:
            entities = json.loads(content)
        except (json.JSONDecodeError, TypeError) as exc:
            logger.warning("entity extraction returned invalid JSON: %s", exc)
            return _fallback(query)

        if not isinstance(entities, dict):
            logger.warning("entity extraction response was not a JSON object")
            return _fallback(query)
        normalised = _normalise_entities(entities, query)
        logger.info(
            "entity extraction: keywords=%r task_type=%s domain_hint=%r year_hint=%r",
            normalised["paper_keywords"],
            normalised["task_type"],
            normalised["domain_hint"],
            normalised["year_hint"],
        )
        return normalised
    except Exception as exc:
        logger.warning("entity extraction failed; using raw query fallback: %s", exc)
        return _fallback(query)
