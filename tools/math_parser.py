"""Math-aware PDF parser that retains a document's heading hierarchy."""

from __future__ import annotations

import io
import re
from collections import Counter
from typing import Any

import pymupdf  # PyMuPDF
import pdfplumber

from config.logging import get_logger

logger = get_logger(__name__)

MATH_SYMBOLS = set("∑∏∫∂∇∞±×÷≤≥≠≈∈∀∃→←↔⊂⊃∪∩αβγδεζηθλμνξπρστυφχψω")
MATH_DENSITY_RE = re.compile(r"[=\+\-*/^_{}\\]|\\frac|\\sum|\\int")
NUMBERED_HEADING_RE = re.compile(r"^(?P<number>\d+(?:\.\d+)*)(?:[.)])?\s+(?P<title>.+)$")
KNOWN_UNNUMBERED_HEADINGS = {
    "abstract", "introduction", "background", "related work", "methods", "method",
    "approach", "experiments", "experiment", "evaluation", "results", "discussion",
    "conclusion", "conclusions", "references", "acknowledgments", "appendix",
}


def _is_math_heavy(text: str, font_flags: int = 0) -> bool:
    if not text.strip():
        return False
    math_chars = sum(1 for char in text if char in MATH_SYMBOLS)
    ratio = (math_chars + len(MATH_DENSITY_RE.findall(text))) / max(len(text), 1)
    return ratio > 0.08 or (bool(font_flags & 2) and ratio > 0.04)


def _heading_level(text: str, font_size: float, body_size: float, flags: int) -> int | None:
    """
    Determine the hierarchical level of a potential section heading.

    The function combines several heuristics to identify whether a text line
    is likely to be a heading:
    - Reject overly long lines or lines ending with sentence punctuation.
    - Detect numbered headings (e.g., "1", "2.3", "4.1.2") and infer the
      hierarchy level from the numbering depth.
    - Recognize common unnumbered headings (e.g., "Abstract", "References").
    - Treat short standalone lines that are bold or have a noticeably larger
      font than the document body as top-level headings.

    Args:
        text: Text content of the PDF line.
        font_size: Font size of the current line.
        body_size: Estimated normal body text font size.
        flags: Font style flags extracted from the PDF (bit 16 indicates bold).

    Returns:
        The detected heading level (1, 2, 3, ...) if the line is classified as
        a heading, otherwise ``None``.
    """
    normalized = " ".join(text.split())

    # Ignore long lines or complete sentences, which are unlikely to be headings.
    if len(normalized) > 160 or normalized.endswith((".", ",", ";", ":")):
        return None

    # Detect numbered headings and determine the level from the number depth.
    # Example:
    #   "1 Introduction"     -> level 1
    #   "2.3 Method"         -> level 2
    #   "4.1.2 Experiment"   -> level 3
    numbered = NUMBERED_HEADING_RE.match(normalized)
    if numbered:
        return numbered.group("number").count(".") + 1

    # Recognize predefined unnumbered section titles.
    lower = normalized.lower()
    if lower in KNOWN_UNNUMBERED_HEADINGS:
        return 1

    # Heuristic for PDF headings:
    # Short standalone lines that are bold or use a noticeably larger font
    # than the document body are treated as top-level headings.
    is_bold = bool(flags & 16)
    is_larger = body_size and font_size >= body_size * 1.15
    if len(normalized.split()) <= 14 and (is_bold or is_larger):
        return 1

    # Not considered a heading.
    return None

def _section_type(heading: str) -> str:
    """Compatibility label only; hierarchy, not this value, drives chunking."""
    text = heading.lower()
    if re.match(r"^(theorem|lemma|proposition|corollary)\b", text):
        return "THEOREM"
    if re.match(r"^proof\b", text):
        return "PROOF"
    if "experiment" in text or "evaluation" in text or "result" in text:
        return "EXPERIMENT"
    if "method" in text or "approach" in text or "model" in text or "architecture" in text:
        return "METHOD"
    if "abstract" == text:
        return "ABSTRACT"
    return "OTHER"


def _path_after_heading(stack: list[tuple[int, str]], level: int, heading: str) -> list[tuple[int, str]]:
    """
    Update the current heading hierarchy after encountering a new heading.

    The function maintains a stack representing the active heading path.
    When a new heading is found:
    - Remove any headings at the same or deeper hierarchy level.
    - Append the new heading as the current active section.

    This preserves the document's hierarchical structure, for example:
        Introduction
          ├── Background
          ├── Motivation
          └── Method
                └── Algorithm

    Example:
        stack = [(1, "Introduction"), (2, "Background")]
        level = 2
        heading = "Method"

        Result:
            [(1, "Introduction"), (2, "Method")]

    Args:
        stack: Current heading path represented as a stack of
            (heading_level, heading_title) tuples.
        level: Hierarchy level of the newly detected heading.
        heading: Text of the new heading.

    Returns:
        The updated heading stack representing the current document path.
    """
    # Remove headings that are siblings or descendants of the new heading.
    while stack and stack[-1][0] >= level:
        stack.pop()

    # Add the new heading to the current hierarchy.
    stack.append((level, heading))

    return stack


def _extract_tables(pdf_bytes: bytes, heading_by_page: dict[int, list[str]]) -> list[dict[str, Any]]:
    """Extract tables from a PDF and preserve their heading hierarchy.

        Each table is converted into a plain-text representation where cells in the
        same row are separated by ``" | "`` and rows are joined with newline
        characters. Every extracted table is returned as a content block enriched
        with metadata describing its page, heading context, and block type.

        Args:
            pdf_bytes: Raw PDF file as bytes.
            heading_by_page: Mapping from page number to the active heading
                hierarchy on that page.

        Returns:
            A list of dictionaries, where each dictionary represents an extracted
            table with the following metadata:

            - ``content``: Text representation of the table.
            - ``section_type``: Always ``"TABLE"``.
            - ``page_num``: Page containing the table.
            - ``is_math``: Always ``False``.
            - ``is_table``: Always ``True``.
            - ``is_heading``: Always ``False``.
            - ``heading_path``: Full heading hierarchy for the page.
            - ``heading``: Innermost heading containing the table.
            - ``heading_level``: Depth of the heading hierarchy.

        Notes:
            If table extraction fails (for example, due to a malformed PDF or an
            unsupported table layout), the exception is logged and an empty list or
            any successfully extracted tables are returned.
        """
    tables: list[dict[str, Any]] = []
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                hierarchy = heading_by_page.get(page_num, [])
                for table in page.extract_tables() or []:
                    rows = [" | ".join(cell or "" for cell in row) for row in table if row]
                    if rows:
                        tables.append({
                            "content": "\n".join(rows), "section_type": "TABLE", "page_num": page_num,
                            "is_math": False, "is_table": True, "is_heading": False,
                            "heading_path": hierarchy, "heading": hierarchy[-1] if hierarchy else "",
                            "heading_level": len(hierarchy),
                        })
    except Exception as exc:
        logger.warning("table extraction failed: %s", exc)
    return tables


def parse_pdf(pdf_bytes: bytes) -> list[dict[str, Any]]:
    """Parse a PDF into ordered content blocks with heading-aware metadata.

    The parser walks through each page, detects headings based on font size and
    formatting, maintains the current heading hierarchy, and annotates every
    text block with its active heading path. Mathematical content is identified
    heuristically, and tables are extracted separately and appended to the
    result.

    Args:
        pdf_bytes: Raw PDF document as bytes.

    Returns:
        A list of dictionaries representing ordered content blocks extracted
        from the PDF. Each block contains:

        - ``content``: Extracted text content.
        - ``section_type``: Logical section category inferred from the heading.
        - ``page_num``: Source page number.
        - ``is_math``: Whether the block appears to contain math-heavy content.
        - ``is_table``: Whether the block represents a table.
        - ``is_heading``: Whether the block is a detected heading.
        - ``heading``: Innermost active heading.
        - ``heading_level``: Depth of the heading hierarchy.
        - ``heading_path``: Full active heading hierarchy.

    Notes:
        The function performs the following steps:

        1. Extract text blocks from each page using PyMuPDF.
        2. Estimate the dominant body-text font size for the page.
        3. Detect headings via :func:`_heading_level`.
        4. Maintain a hierarchical heading stack across blocks.
        5. Annotate regular text blocks with the current heading context.
        6. Extract tables using :func:`_extract_tables`.
        7. Return all collected blocks in reading order, with tables appended
           after text extraction.

    Raises:
        This function does not explicitly raise parsing errors during normal
        operation. The underlying PDF library may raise exceptions if the input
        bytes do not represent a valid PDF document.
    """
    blocks: list[dict[str, Any]] = []
    heading_stack: list[tuple[int, str]] = []
    heading_by_page: dict[int, list[str]] = {}
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    try:
        for page_num, page in enumerate(doc, start=1):
            page_dict = page.get_text("dict")
            sizes = [span.get("size", 0.0) for block in page_dict.get("blocks", []) if block.get("type") == 0
                     for line in block.get("lines", []) for span in line.get("spans", []) if span.get("text", "").strip()]
            body_size = Counter(round(size, 1) for size in sizes).most_common(1)[0][0] if sizes else 0.0

            for block in page_dict.get("blocks", []):
                if block.get("type") != 0:
                    continue
                spans = [span for line in block.get("lines", []) for span in line.get("spans", []) if span.get("text", "").strip()]
                text = " ".join(span["text"] for span in spans).strip()
                if not text:
                    continue
                max_flags = max((span.get("flags", 0) for span in spans), default=0)
                max_size = max((span.get("size", 0.0) for span in spans), default=0.0)
                level = _heading_level(text, max_size, body_size, max_flags)

                if level is not None:
                    heading_stack = _path_after_heading(heading_stack, level, text)
                    path = [item[1] for item in heading_stack]
                    blocks.append({
                        "content": text, "section_type": _section_type(text), "page_num": page_num,
                        "is_math": False, "is_table": False, "is_heading": True,
                        "heading": text, "heading_level": level, "heading_path": path,
                    })
                    heading_by_page[page_num] = path.copy()
                    continue

                path = [item[1] for item in heading_stack]
                is_math = _is_math_heavy(text, max_flags)
                blocks.append({
                    "content": text, "section_type": _section_type(path[-1] if path else ""),
                    "page_num": page_num, "is_math": is_math, "is_table": False, "is_heading": False,
                    "heading": path[-1] if path else "", "heading_level": len(path), "heading_path": path,
                })
                heading_by_page.setdefault(page_num, path.copy())
    finally:
        doc.close()

    blocks.extend(_extract_tables(pdf_bytes, heading_by_page))
    logger.info("parsed PDF into %d blocks", len(blocks))
    return blocks