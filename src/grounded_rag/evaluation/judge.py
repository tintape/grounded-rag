"""LLM-as-judge scoring (RAGAS-style faithfulness and answer correctness).

Implemented directly instead of depending on `ragas`, whose import chain breaks with current
langchain releases. Two scores in [0, 1]:
- faithfulness: share of the answer's claims supported by the cited passages;
- correctness: agreement with the reference answer (None when no reference is given).
"""

import json
import logging
import re
from dataclasses import dataclass
from typing import Protocol

logger = logging.getLogger(__name__)

JUDGE_SYSTEM = """You grade answers from a document-grounded support bot. Reply with JSON only:
{"faithfulness": <0..1>, "correctness": <0..1 or null>}
faithfulness = fraction of factual claims in ANSWER that are directly supported by PASSAGES.
correctness = how well ANSWER matches REFERENCE in meaning (1 = same facts, 0 = wrong/missing);
use null when REFERENCE is "none"."""

_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


class Generator(Protocol):
    def generate(self, system: str, user: str) -> str: ...


@dataclass(frozen=True, slots=True)
class JudgeScores:
    faithfulness: float | None
    correctness: float | None


def _clamp(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return min(1.0, max(0.0, float(value)))


def parse_scores(raw: str) -> JudgeScores:
    match = _JSON_OBJECT.search(raw)
    if not match:
        return JudgeScores(None, None)
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        logger.warning("Judge returned invalid JSON: %s", raw[:200])
        return JudgeScores(None, None)
    return JudgeScores(_clamp(data.get("faithfulness")), _clamp(data.get("correctness")))


class LLMJudge:
    def __init__(self, generator: Generator) -> None:
        self._generator = generator

    def score(
        self, question: str, answer: str, passages: tuple[str, ...], reference: str | None
    ) -> JudgeScores:
        context = "\n\n".join(f"- {p}" for p in passages)
        user = (
            f"QUESTION: {question}\n\nPASSAGES:\n{context}\n\nANSWER: {answer}\n\n"
            f"REFERENCE: {reference or 'none'}"
        )
        try:
            return parse_scores(self._generator.generate(JUDGE_SYSTEM, user))
        except Exception:
            logger.exception("Judge call failed")
            return JudgeScores(None, None)
