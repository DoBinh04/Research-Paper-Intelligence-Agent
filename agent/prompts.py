"""Centralized prompts for the research paper agent."""

QUERY_REWRITE_PROMPT = """Rewrite the user's research question into a precise academic search query.
Focus on: method names, benchmarks, metrics, model architectures, and comparison terms.
Return ONLY the rewritten query, no explanation.

Examples:
- "LoRA vs adapter which is better?" -> "LoRA vs Adapter layers parameter efficiency GLUE benchmark comparison"
- "transformer paper attention" -> "scaled dot-product self-attention transformer architecture Vaswani 2017"
"""

ENTITY_EXTRACTION_PROMPT = """You are an expert academic paper search assistant. Extract structured technical entities from the user's natural-language research query.

Return ONLY a valid JSON object with exactly these keys: paper_keywords, task_type, domain_hint, year_hint.

Rules:
- paper_keywords must be a JSON list of at most 6 technical search terms. Extract method names, model names, and technical concepts. Expand abbreviations when useful (for example, LoRA must also include "low-rank adaptation").
- task_type must be exactly one of: "find_paper", "compare", "summarize", "explain".
- domain_hint must be a field of study when detectable (for example, "NLP" or "CV"); otherwise use null.
- year_hint must be an integer when the user mentions a time range; otherwise use null.
- If information is not present, use null, never Python None.
- Zero tolerance for markdown fences, explanations, or any text outside the JSON object.

Example:
Input: "attention mechanism in transformers vs RNN"
Output: {"paper_keywords": ["attention mechanism", "transformer", "self-attention", "RNN", "sequence modeling"], "task_type": "compare", "domain_hint": "NLP", "year_hint": null}
"""

SYNTHESIZE_PROMPT = """You are synthesizing a literature review answer from retrieved paper excerpts.

User question:
{query}

Retrieved context:
{context}

Citation graph summary:
{graph_summary}

Detected conflicts:
{conflicts}

Write a structured markdown report with these sections:
## Summary
## Key Findings
## Experimental Comparison
## Conflicts
## References

Rules:
- Cite sources as [paper title, heading path, chunk index] inline
- Compare benchmark numbers when available
- Note agreement and disagreement between papers
- Be factual — only use provided context
"""

CONFLICT_DETECT_PROMPT = """Given two experimental claims from different papers, determine if they truly conflict.
Return JSON: {{"is_conflict": true/false, "explanation": "..."}}
"""

OFF_TOPIC_GUARD = """Determine if the query is about ML/AI/math research papers.
Return YES or NO only.
"""
