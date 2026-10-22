"""Run an evaluation set through the answerer and compute quality metrics."""

import logging
import math
import statistics
import time
from dataclasses import dataclass
from typing import Protocol

from grounded_rag.evaluation.dataset import EvalCase
from grounded_rag.evaluation.judge import JudgeScores
from grounded_rag.models import Answer

logger = logging.getLogger(__name__)


class Answerer(Protocol):
    def answer(self, question: str) -> Answer: ...


class Judge(Protocol):
    def score(
        self, question: str, answer: str, passages: tuple[str, ...], reference: str | None
    ) -> JudgeScores: ...


@dataclass(frozen=True, slots=True)
class EvalResult:
    case: EvalCase
    answer: Answer
    refusal_correct: bool
    source_hit: bool | None
    scores: JudgeScores
    latency_s: float
    error: str | None = None

    @property
    def judged(self) -> bool:
        return not self.answer.refused and self.error is None


@dataclass(frozen=True, slots=True)
class EvalSummary:
    results: tuple[EvalResult, ...]

    def _rate(self, flags: list[bool]) -> float | None:
        return round(sum(flags) / len(flags), 3) if flags else None

    @property
    def refusal_accuracy(self) -> float | None:
        return self._rate([r.refusal_correct for r in self.results])

    @property
    def answered_rate(self) -> float | None:
        return self._rate([r.refusal_correct for r in self.results if r.case.answerable])

    @property
    def refused_rate(self) -> float | None:
        return self._rate([r.refusal_correct for r in self.results if not r.case.answerable])

    @property
    def source_hit_rate(self) -> float | None:
        return self._rate([r.source_hit for r in self.results if r.source_hit is not None])

    @property
    def errors(self) -> int:
        return sum(1 for r in self.results if r.error)

    @property
    def judged_count(self) -> int:
        return sum(1 for r in self.results if r.judged)

    @property
    def judge_failures(self) -> int:
        return sum(1 for r in self.results if r.judged and r.scores.faithfulness is None)

    @property
    def hallucinated(self) -> int:
        """Unanswerable questions that got an answer instead of a refusal."""
        return sum(1 for r in self.results if not r.case.answerable and r.judged)

    def _mean(self, values: list[float | None]) -> float | None:
        present = [v for v in values if v is not None]
        return round(statistics.fmean(present), 3) if present else None

    @property
    def faithfulness(self) -> float | None:
        return self._mean([r.scores.faithfulness for r in self.results])

    @property
    def correctness(self) -> float | None:
        return self._mean([r.scores.correctness for r in self.results])

    @property
    def p95_latency_s(self) -> float | None:
        latencies = sorted(r.latency_s for r in self.results)
        if not latencies:
            return None
        nearest_rank = math.ceil(0.95 * len(latencies)) - 1
        return round(latencies[nearest_rank], 2)


def _source_hit(case: EvalCase, answer: Answer) -> bool | None:
    if not case.answerable or not case.expected_sources:
        return None
    cited = {(s.doc_name, s.page) for s in answer.sources}
    return any((e.doc_name, e.page) in cited for e in case.expected_sources)


def _evaluate(case: EvalCase, answerer: Answerer, judge: Judge | None) -> EvalResult:
    started = time.perf_counter()
    try:
        answer = answerer.answer(case.question)
    except Exception as exc:  # one failing case must not abort the whole run
        logger.exception("Case %s failed", case.id)
        return EvalResult(
            case=case,
            answer=Answer.refusal(f"Error: {exc}"),
            refusal_correct=False,
            source_hit=False if _source_hit(case, Answer.refusal()) is not None else None,
            scores=JudgeScores(None, None),
            latency_s=round(time.perf_counter() - started, 3),
            error=str(exc),
        )
    latency = time.perf_counter() - started
    scores = JudgeScores(None, None)
    # Judge every produced answer, including answers to unanswerable questions (hallucinations).
    if judge is not None and not answer.refused:
        passages = tuple(s.excerpt for s in answer.sources)
        scores = judge.score(case.question, answer.answer, passages, case.reference_answer)
    return EvalResult(
        case=case,
        answer=answer,
        refusal_correct=answer.refused != case.answerable,
        source_hit=_source_hit(case, answer),
        scores=scores,
        latency_s=round(latency, 3),
    )


def run_eval(
    cases: tuple[EvalCase, ...], answerer: Answerer, judge: Judge | None = None
) -> EvalSummary:
    return EvalSummary(results=tuple(_evaluate(c, answerer, judge) for c in cases))
