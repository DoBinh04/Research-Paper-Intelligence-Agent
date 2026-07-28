"""Detect conflicting numeric claims across experiment sections."""

from __future__ import annotations

import itertools
import re
from typing import List

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from agent.prompts import CONFLICT_DETECT_PROMPT
from config.logging import get_logger
from config.settings import settings

logger = get_logger(__name__)

METRIC_RE = re.compile(
    r"""
    (?P<metric>
        accuracy
        |acc
        |f1(?:-score)?
        |precision
        |recall
        |bleu
        |rouge(?:-?[12l])?
        |meteor
        |cider
        |spice
        |perplexity|ppl
        |loss
        |auc|roc-auc|pr-auc
        |map
        |mrr
        |ndcg(?:@\d+)?
        |hit(?:s)?@\d+
        |recall@\d+
        |precision@\d+
        |exact\s*match|em
    )
    (?:\s*(?:score)?\s*[:=]?\s*|\s+.{0,25}?)
    (?P<value>\d+(?:\.\d+)?)
    \s*%?
    """,
    re.I | re.X,
)
DATASET_RE = re.compile(
    r"""
    \b(
        GLUE
        |SuperGLUE
        |SQuAD(?:\s*2\.0)?
        |ImageNet
        |CIFAR(?:-10|-100)?
        |MNIST
        |Fashion-?MNIST
        |WMT
        |CoLA
        |MNLI
        |QQP
        |QNLI
        |RTE
        |STS-?B
        |SST-?2
        |BoolQ
        |WikiText(?:-103)?
        |MS\s*MARCO
        |Natural\s*Questions
        |HotpotQA
    )\b
    """,
    re.I | re.X,
)
THRESHOLD = 2.0
EXPERIMENT_HEADING_RE = re.compile(r"\b(experiment|evaluation|result|benchmark|ablation)\b", re.I)
_SENTENCE_RE = re.compile(
    r"(?<=[.!?])\s+|\n{2,}"
)

def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_RE.split(text) if s.strip()]

def _find_dataset(sentence: str) -> str | None:
    m = DATASET_RE.search(sentence)
    if m:
        return m.group(0).upper()
    return None

def _normalize_metric(metric: str) -> str:
    metric = metric.lower().strip()

    aliases = {
        "acc": "accuracy",
        "f1-score": "f1",
        "ppl": "perplexity",
        "roc-auc": "auc",
        "pr-auc": "auc",
        "em": "exact match",
    }

    return aliases.get(metric, metric)

def _extract_claims(text: str, paper_id: str) -> list[dict]:

    claims = []

    current_dataset = "GENERAL"

    for sentence in _split_sentences(text):

        dataset = _find_dataset(sentence)

        if dataset:
            current_dataset = dataset

        for m in METRIC_RE.finditer(sentence):

            claims.append(
                {
                    "paper_id": paper_id,
                    "metric": _normalize_metric(
                        m.group("metric")
                    ),
                    "value": float(m.group("value")),
                    "dataset": current_dataset,
                    "raw": sentence,
                }
            )

    return claims


def detect_conflicts(paper_chunks: List[dict], topic: str = "") -> List[dict]:
    experiment_chunks = [
        c for c in paper_chunks
        if (c.get("section_type") or "").upper() in ("EXPERIMENT", "TABLE")
        or EXPERIMENT_HEADING_RE.search(c.get("heading_path") or c.get("section_heading") or "")
    ]
    if topic:
        topic_upper = topic.upper()
        experiment_chunks = [
            c for c in experiment_chunks if topic_upper in c.get("content", "").upper()
        ] or experiment_chunks

    all_claims: list[dict] = []
    for chunk in experiment_chunks:
        all_claims.extend(_extract_claims(chunk.get("content", ""), chunk.get("paper_id", "?")))

    conflicts: list[dict] = []
    grouped: dict[tuple[str, str], list[dict]] = {}
    for claim in all_claims:
        key = (claim["dataset"], claim["metric"].lower())
        grouped.setdefault(key, []).append(claim)

    for (_, metric), claims in grouped.items():
        if len(claims) < 2:
            continue
        for a, b in itertools.combinations(claims, 2): #generate all pairs of paper
            if a["paper_id"] == b["paper_id"]:
                continue
            delta = abs(a["value"] - b["value"])
            if delta >= THRESHOLD:
                conflict = {
                    "paper_a": a["paper_id"],
                    "paper_b": b["paper_id"],
                    "claim_a": f"{a['value']}%",
                    "claim_b": f"{b['value']}%",
                    "metric": f"{metric} on {a['dataset']}",
                    "delta": delta,
                }
                conflicts.append(conflict)

    if settings.openai_api_key and conflicts:
        conflicts = _llm_verify(conflicts[:5])

    logger.info("detected %d conflicts", len(conflicts))
    return conflicts


def _llm_verify(conflicts: list[dict]) -> list[dict]:
    llm = ChatOpenAI(model=settings.llm_model, temperature=0, api_key=settings.openai_api_key)
    verified: list[dict] = []
    for c in conflicts:
        prompt = (
            f"{CONFLICT_DETECT_PROMPT}\n\n"
            f"Paper A ({c['paper_a']}): {c['claim_a']}\n"
            f"Paper B ({c['paper_b']}): {c['claim_b']}\n"
            f"Metric: {c['metric']}"
        )
        try:
            resp = llm.invoke([SystemMessage(content="Verify conflicts."), HumanMessage(content=prompt)])
            text = (resp.content or "").lower()
            if "true" in text or "is_conflict" in text:
                verified.append(c)
        except Exception:
            verified.append(c)
    return verified or conflicts