"""Structured JSON logging for observability."""

import logging
import sys
from typing import Optional

try:
    from pythonjsonlogger import jsonlogger
except ImportError:  # pragma: no cover
    jsonlogger = None

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Create and configure a logger instance.

    The logger outputs logs to standard output (stdout). If the
    python-json-logger package is installed, logs are formatted as
    structured JSON; otherwise a standard text formatter is used.

    Existing loggers are reused to avoid attaching duplicate handlers.

    Args:
        name: Logger name, typically __name__.
        level: Minimum logging level (default: INFO).

    Returns:
        Configured Logger instance.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    handler = logging.StreamHandler(sys.stdout)
    if jsonlogger:
        formatter = jsonlogger.JsonFormatter(
            "%(timestamp)s %(level)s %(name)s %(message)s",
            rename_fields={"levelname": "level", "name": "module"},
        )
    else:
        formatter = logging.Formatter(
            "%(asctime)s %(levelname)s [%(name)s] %(message)s"
        )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


def log_tool_call(
    logger: logging.Logger,
    tool: str,
    latency_ms: float,
    paper_id: Optional[str] = None,
    **extra: object,
) -> None:
    """
    Log tool execution metadata in a structured format.

    Records the tool name, execution latency, optional paper ID,
    and any additional metadata provided through keyword arguments.

    Args:
        logger: Logger instance used for logging.
        tool: Name of the executed tool.
        latency_ms: Execution time in milliseconds.
        paper_id: Optional paper identifier associated with the call.
        **extra: Additional metadata fields to include in the log.

    Returns:
        None
    """
    payload = {"tool": tool, "latency_ms": round(latency_ms, 2), **extra}
    if paper_id:
        payload["paper_id"] = paper_id
    logger.info("tool_call", extra=payload)
