"""LLM-based query rewriting for better retrieval."""

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from agent.prompts import QUERY_REWRITE_PROMPT
from config.logging import get_logger
from config.settings import settings

logger = get_logger(__name__)


def rewrite_query(query: str) -> str:
    """ Rewrite a user query into a retrieval-optimized academic search query.

    Uses an LLM to transform a natural-language research question into a
    concise query containing important academic keywords such as method names,
    model architectures, benchmarks, evaluation metrics, and comparison terms.
    The rewritten query is intended to improve semantic retrieval quality in
    the RAG pipeline.

    If no OpenAI API key is configured or the rewrite request fails, the
    original query is returned unchanged so the retrieval pipeline can
    continue without interruption.

    Args:
        query: Original user research question.

    Returns:
        The rewritten academic search query, or the original query if
        rewriting is unavailable or fails.
    """
    if not settings.openai_api_key:
        logger.warning("No OPENAI_API_KEY — skipping query rewrite")
        return query

    llm = ChatOpenAI(model=settings.llm_model, temperature=0, api_key=settings.openai_api_key, base_url=settings.openai_base_url)
    messages = [
        SystemMessage(content=QUERY_REWRITE_PROMPT),
        HumanMessage(content=query),
    ]
    try:
        response = llm.invoke(messages)
        rewritten = (response.content or query).strip()
        logger.info("query re   written: %r -> %r", query[:80], rewritten[:80])
        return rewritten
    except Exception as exc:
        logger.error("query rewrite failed: %s", exc)
        return query