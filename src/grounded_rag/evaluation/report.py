"""Write evaluation results as JSON (machine) and Markdown (human)."""

import json
from pathlib import Path

from grounded_rag.evaluation.runner import EvalResult, EvalSummary


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.0%}"


def _num(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def summary_dict(summary: EvalSummary) -> dict:
    return {
        "cases": len(summary.results),
        "refusal_accuracy": summary.refusal_accuracy,
        "answerable_answered_rate": summary.answered_rate,
        "unanswerable_refused_rate": summary.refused_rate,
        "source_hit_rate": summary.source_hit_rate,
        "faithfulness": summary.faithfulness,
        "correctness": summary.correctness,
        "p95_latency_s": summary.p95_latency_s,
        "answers_judged": summary.judged_count,
        "judge_failures": summary.judge_failures,
        "unanswerable_answered": summary.hallucinated,
        "errors": summary.errors,
    }


def _result_dict(result: EvalResult) -> dict:
    return {
        "id": result.case.id,
        "question": result.case.question,
        "answerable": result.case.answerable,
        "answer": result.answer.answer,
        "refused": result.answer.refused,
        "sources": [f"{s.doc_name} p.{s.page}" for s in result.answer.sources],
        "refusal_correct": result.refusal_correct,
        "source_hit": result.source_hit,
        "faithfulness": result.scores.faithfulness,
        "correctness": result.scores.correctness,
        "latency_s": result.latency_s,
        "error": result.error,
    }


def _mark(flag: bool | None) -> str:
    return {True: "✅", False: "❌", None: "—"}[flag]


def to_markdown(summary: EvalSummary) -> str:
    s = summary_dict(summary)
    lines = [
        "# Evaluation report",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Questions | {s['cases']} |",
        f"| Refusal accuracy (answered when answerable, refused when not) | {_pct(s['refusal_accuracy'])} |",  # noqa: E501
        f"| Answerable questions answered | {_pct(s['answerable_answered_rate'])} |",
        f"| Unanswerable questions refused | {_pct(s['unanswerable_refused_rate'])} |",
        f"| Correct document + page cited | {_pct(s['source_hit_rate'])} |",
        f"| Faithfulness (LLM judge, 0-1) | {_num(s['faithfulness'])} |",
        f"| Correctness vs reference (LLM judge, 0-1) | {_num(s['correctness'])} |",
        f"| Unanswerable questions answered anyway | {s['unanswerable_answered']} |",
        f"| p95 latency | {_num(s['p95_latency_s'])} s |",
        f"| Errors (service failures) | {s['errors']} |",
        "",
        f"Judge scores average over the {s['answers_judged']} answers the bot produced "
        f"({s['judge_failures']} judge failures excluded); refused questions are not scored. "
        "The judge is an LLM, typically the same model as the generator — treat its scores as "
        "indicative and spot-check the per-question table.",
        "",
        "## Per question",
        "",
        "| ID | Question | Behaviour | Source | Faithful | Correct | Cited |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in summary.results:
        cited = ", ".join(f"{x.doc_name} p.{x.page}" for x in r.answer.sources) or "—"
        lines.append(
            f"| {r.case.id} | {r.case.question} | {_mark(r.refusal_correct)} | "
            f"{_mark(r.source_hit)} | {_num(r.scores.faithfulness)} | "
            f"{_num(r.scores.correctness)} | {cited} |"
        )
    return "\n".join(lines) + "\n"


def write_report(summary: EvalSummary, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path, md_path = out_dir / "report.json", out_dir / "report.md"
    payload = {
        "summary": summary_dict(summary),
        "results": [_result_dict(r) for r in summary.results],
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(to_markdown(summary), encoding="utf-8")
    return json_path, md_path
