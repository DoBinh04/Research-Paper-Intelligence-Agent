"""Session memory — cache fetched papers and query history."""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Optional


class SessionMemory:
    """In-memory session storage for papers and query history.

    Stores temporary data for each user session, including fetched papers
    and previous queries. Data is kept only for the lifetime of the process.
    """
    def __init__(self) -> None:
        self._papers: dict[str, dict[str, Any]] = {} #dict: key is session_id, value is paper_id
        self._queries: dict[str, list[str]] = {} #dict: key is session_id, value is query
        self._timestamps: dict[str, float] = {}

    def get_paper(self, session_id: str, paper_id: str) -> Optional[dict]:
        """Return a cached paper for the given session, if available."""
        session = self._papers.get(session_id, {})
        return session.get(paper_id)

    def set_paper(self, session_id: str, paper_id: str, paper_data: dict) -> None:
        """Cache a paper under the specified session."""
        self._papers.setdefault(session_id, {})[paper_id] = paper_data

    def list_papers(self, session_id: str) -> list[dict]:
        """Return all cached papers for a session."""
        return list(self._papers.get(session_id, {}).values())

    def add_query(self, session_id: str, query: str) -> None:
        """Append a user query to the session history."""
        self._queries.setdefault(session_id, []).append(query)

    def get_queries(self, session_id: str) -> list[str]:
        """Return the query history for a session."""
        return self._queries.get(session_id, [])

    def reset(self, session_id: str) -> None:
        """Remove all cached data associated with a session."""
        self._papers.pop(session_id, None)
        self._queries.pop(session_id, None)
        self._timestamps.pop(session_id, None)


@lru_cache(maxsize=1)
def get_memory() -> SessionMemory:
    """Return the shared SessionMemory instance."""
    return SessionMemory()