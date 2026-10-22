import json
from dataclasses import dataclass, field

import pytest
from pydantic import ValidationError

from grounded_rag.evaluation.dataset import load_cases, parse_cases
from grounded_rag.evaluation.judge import JudgeScores, LLMJudge, parse_scores
from grounded_rag.evaluation.report import write_report
from grounded_rag.evaluation.runner import run_eval
from grounded_rag.models import Answer, Source

CASES_JSON = json.dumps(
    [
        {
            "id": "reset",
            "question": "How do I reset?",
            "reference_answer": "Hold Home and Spot for 10 seconds.",
            "expected_sources": [{"doc_name": "guide.pdf", "page": 4}],
        },
        {
            "id": "warranty",
            "question": "Battery warranty?",
            "expected_sources": [{"doc_name": "warranty.pdf", "page": 2}],
        },
        {"id": "ceo", "question": "Who is the CEO?", "answerable": False},
        {"id": "price", "question": "What does it cost?", "answerable": False},
    ]
)


def grounded(doc: str, page: int) -> Answer:
    return Answer("Answer [1].", False, (Source(doc, page, "excerpt", 0.8),))


@dataclass
class ScriptedAnswerer:
    replies: dict[str, Answer]

    def answer(self, question: str) -> Answer:
        return self.replies[question]


@dataclass
class FakeJudge:
    calls: list[str] = field(default_factory=list)

    def score(self, question, answer, passages, reference):
        self.calls.append(question)
        return JudgeScores(faithfulness=1.0, correctness=0.5 if reference else None)


@pytest.fixture
def summary():
    answerer = ScriptedAnswerer(
        {
            "How do I reset?": grounded("guide.pdf", 4),
            "Battery warranty?": grounded("faq.pdf", 1),  # answered, wrong page
            "Who is the CEO?": Answer.refusal(),
            "What does it cost?": grounded("faq.pdf", 3),  # should have refused
        }
    )
    judge = FakeJudge()
    result = run_eval(parse_cases(CASES_JSON), answerer, judge)
    return result, judge


def test_refusal_metrics(summary):
    result, _ = summary
    assert result.refusal_accuracy == 0.75
    assert result.answered_rate == 1.0
    assert result.refused_rate == 0.5


def test_source_hit_rate_counts_only_cases_with_expected_sources(summary):
    result, _ = summary
    assert result.source_hit_rate == 0.5


def test_judge_runs_only_on_answered_answerable_cases(summary):
    result, judge = summary
    assert judge.calls == ["How do I reset?", "Battery warranty?"]
    assert result.faithfulness == 1.0
    assert result.correctness == 0.5


def test_report_files_written(summary, tmp_path):
    result, _ = summary
    json_path, md_path = write_report(result, tmp_path)
    data = json.loads(json_path.read_text())
    assert data["summary"]["cases"] == 4
    assert len(data["results"]) == 4
    markdown = md_path.read_text()
    assert "Refusal accuracy" in markdown
    assert "| price |" in markdown


def test_load_cases_from_file(tmp_path):
    path = tmp_path / "q.json"
    path.write_text(CASES_JSON)
    assert len(load_cases(path)) == 4


def test_duplicate_ids_rejected():
    raw = json.dumps([{"id": "a", "question": "x"}, {"id": "a", "question": "y"}])
    with pytest.raises(ValueError, match="Duplicate"):
        parse_cases(raw)


def test_unanswerable_with_expected_sources_rejected():
    raw = json.dumps(
        [
            {
                "id": "a",
                "question": "x",
                "answerable": False,
                "expected_sources": [{"doc_name": "a.pdf", "page": 1}],
            }
        ]
    )
    with pytest.raises(ValidationError):
        parse_cases(raw)


@pytest.mark.parametrize(
    "raw,expected",
    [
        ('{"faithfulness": 0.9, "correctness": 1}', JudgeScores(0.9, 1.0)),
        ('Sure! {"faithfulness": 1.7, "correctness": null}', JudgeScores(1.0, None)),
        ('{"faithfulness": "high"}', JudgeScores(None, None)),
        ("no json here", JudgeScores(None, None)),
        ("{broken json}", JudgeScores(None, None)),
    ],
)
def test_parse_scores_is_defensive(raw, expected):
    assert parse_scores(raw) == expected


def test_llm_judge_swallows_upstream_errors_into_empty_scores():
    class Broken:
        def generate(self, system, user):
            raise TimeoutError

    assert LLMJudge(Broken()).score("q", "a", ("p",), None) == JudgeScores(None, None)


def test_llm_judge_sends_passages_and_reference():
    seen = {}

    class Recorder:
        def generate(self, system, user):
            seen["user"] = user
            return '{"faithfulness": 1, "correctness": 1}'

    scores = LLMJudge(Recorder()).score("q", "ans", ("passage one",), "ref")
    assert scores == JudgeScores(1.0, 1.0)
    assert "passage one" in seen["user"] and "REFERENCE: ref" in seen["user"]
