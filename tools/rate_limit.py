from __future__ import annotations

import asyncio
import threading
import time
from email.utils import parsedate_to_datetime
from typing import Mapping


class AsyncRateLimiter:
    """Rate limiter theo khoảng cách tối thiểu giữa các request."""

    def __init__(self, min_interval_seconds: float) -> None:
        self._min_interval = min_interval_seconds
        self._next_allowed_at = 0.0
        self._lock = threading.Lock()

    async def wait(self) -> None:
        # threading.Lock giúp limiter dùng được cả khi project tạo nhiều event loop/thread.
        with self._lock:
            now = time.monotonic()
            scheduled_at = max(now, self._next_allowed_at)
            self._next_allowed_at = scheduled_at + self._min_interval

        delay = scheduled_at - time.monotonic()
        if delay > 0:
            await asyncio.sleep(delay)

    def defer(self, seconds: float) -> None:
        """Áp dụng thời gian chờ do server yêu cầu, ví dụ Retry-After."""
        with self._lock:
            self._next_allowed_at = max(
                self._next_allowed_at,
                time.monotonic() + max(0.0, seconds),
            )


# These are process-wide on purpose: calls from paper search and citation
# traversal consume the same upstream API quota.
SEMANTIC_SCHOLAR_LIMITER = AsyncRateLimiter(1.05)
ARXIV_LIMITER = AsyncRateLimiter(0.34)


def retry_after_seconds(headers: Mapping[str, str], fallback: float) -> float:
    """Return the server's Retry-After delay, or a safe fallback."""
    value = headers.get("retry-after")
    if not value:
        return fallback
    try:
        return max(0.0, float(value))
    except ValueError:
        try:
            return max(0.0, parsedate_to_datetime(value).timestamp() - time.time())
        except (TypeError, ValueError, IndexError, OverflowError):
            return fallback
